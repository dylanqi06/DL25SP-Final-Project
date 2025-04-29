from typing import List
import numpy as np
from torch import nn
import torch


def build_mlp(layers_dims: List[int]):
    """
    Builds a simple MLP given a list of layer dimensions.
    """
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class MockModel(torch.nn.Module):
    """
    Does nothing; returns random embeddings for testing.
    """
    def __init__(self, device="cuda", output_dim=256):
        super().__init__()
        self.device = device
        self.repr_dim = output_dim

    def forward(self, states, actions):
        """
        Args:
            states: [B, T, Ch, H, W]
            actions: [B, T-1, 2]

        Returns:
            Random embeddings: [B, T, repr_dim]
        """
        B, T, _ = actions.shape
        return torch.randn((B, T + 1, self.repr_dim), device=self.device)


class Encoder(nn.Module):
    """
    Convolutional encoder that maps image frames to embeddings.
    """
    def __init__(self, input_shape=(2, 65, 65), embedding_dim=256):
        super().__init__()
        C, H, W = input_shape
        self.cnn = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, embedding_dim),
        )

    def forward(self, x):  # x: [B, C, H, W]
        out = self.cnn(x)
        return out


# class Predictor(nn.Module):
#     """
#     MLP that predicts next-step embeddings from current embedding + action.
#     """
#     def __init__(self, embedding_dim=256, action_dim=2):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(embedding_dim + action_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, embedding_dim),
#         )

#     def forward(self, z, action):
#         x = torch.cat([z, action], dim=-1)
#         return self.model(x)

class Predictor(nn.Module):
    def __init__(self, embedding_dim=256, action_dim=2, num_layers=2, nhead=4, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        # self.input_dim = embedding_dim + action_dim
        self.input_dim = 256 
        self.input_proj = nn.Linear(embedding_dim + action_dim, self.input_dim)

        self.pos_encoding = nn.Parameter(torch.randn(100, self.input_dim))  # supports up to 100 steps
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_dim, nhead=nhead, dim_feedforward=embedding_dim * 2, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(self.input_dim, embedding_dim)

    def forward(self, z_seq, a_seq):
        """
        z_seq: [B, T, D]
        a_seq: [B, T, A]
        """
        x = torch.cat([z_seq, a_seq], dim=-1)  # [B, T, D+A]
        x = self.input_proj(x) 
        x = x + self.pos_encoding[:x.size(1)]  # add position encodings

        x = x.transpose(0, 1)  # transformer expects [T, B, D]
        x = self.transformer(x)  # [T, B, D+A]
        x = x.transpose(0, 1)  # [B, T, D+A]

        return self.fc_out(x)  # [B, T, D]


class JEPA(nn.Module):
    """
    Joint Embedding Prediction Architecture.
    Predicts future frame embeddings given past frames and actions.
    """
    def __init__(
        self,
        input_shape=(2, 65, 65),
        embedding_dim=256,
        action_dim=2,
        ema_decay=0.99,
        var_w=1.0,
        cov_w=0.01,
        gamma=1.0,
    ):
        super().__init__()
        C, H, W = input_shape
        # print(f"[JEPA.__init__] input_shape={input_shape}, in_ch={C}")
        self.encoder = Encoder(input_shape=input_shape, embedding_dim=embedding_dim)
        self.target_encoder = Encoder(input_shape=input_shape, embedding_dim=embedding_dim)
        self.predictor = Predictor(embedding_dim=embedding_dim, action_dim=action_dim, num_layers=2, nhead=4, dropout=0.1)
        self.ema_decay = ema_decay
        self._initialize_target()

        self.repr_dim = embedding_dim
        self.var_w, self.cov_w, self.gamma = var_w, cov_w, gamma

    def _initialize_target(self):
        for param_q, param_k in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    def update_target(self):
        for param_q, param_k in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            param_k.data = (
                param_k.data * self.ema_decay + param_q.data * (1.0 - self.ema_decay)
            )

    def forward(self, states, actions):
        # states: [B, T_states, C, H, W], actions: [B, T_actions, action_dim]
        B, T_states, C, H, W = states.shape
        flat = states.view(-1, C, H, W)   #[B*T, 2, 65, 65] 
        zs = self.encoder(flat).view(B, T_states, -1)  #[B, T, D] 

        if T_states > 1:
            z_in_seq = zs[:, :-1, :]      # [B, T-1, D]  # use frames 0 to T-2
            a_in_seq = actions             # [B, T-1, A]

            preds = self.predictor(z_in_seq, a_in_seq)  # [B, T-1, D]

            with torch.no_grad():
                targ_z = self.target_encoder(flat).view(B, T_states, -1)[:, 1:]  # [B, T-1, D]

        else:
            T_actions = actions.size(1)  # 16
            z_prev = zs[:, 0]             # [B, D]

            preds = [z_prev]              # first prediction is just initial embedding
            for t in range(T_actions):
                a_t = actions[:, t]       # [B, A]
                p = self.predictor(preds[-1].unsqueeze(1), a_t.unsqueeze(1))  # feed single timestep
                preds.append(p.squeeze(1))  # append [B, D]

            preds = torch.stack(preds, dim=1)  # [B, 17, D] zs.size(-1), device=zs.device)  # empty for probing mode
            # print(f"[JEPA.forward] preds.shape={preds.shape}")
        if self.training:
            return preds, targ_z
        else:
            return preds


class Prober(nn.Module):
    """
    MLP probe that predicts target attributes from embeddings.
    """
    def __init__(self, embedding: int, arch: str, output_shape: List[int]):
        super().__init__()
        # print(f"[Prober.__init__] embedding={embedding}, arch='{arch}', output_shape={output_shape}")
        self.output_dim = int(np.prod(output_shape))
        arch_list = list(map(int, arch.split("-"))) if arch else []
        dims = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.prober = nn.Sequential(*layers)

    def forward(self, e):
        # print(f"[Prober.forward] input e.shape={e.shape}")
        out = self.prober(e)
        # print(f"[Prober.forward] output out.shape={out.shape}")
        return out
