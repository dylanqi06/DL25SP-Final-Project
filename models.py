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
        # print(f"[Encoder.__init__] in_ch={C}")
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
        # print(f"[Encoder.forward] x.shape={x.shape}")
        out = self.cnn(x)
        # print(f"[Encoder.forward] out.shape={out.shape}")
        return out


class Predictor(nn.Module):
    """
    MLP that predicts next-step embeddings from current embedding + action.
    """
    def __init__(self, embedding_dim=256, action_dim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, z, action):
        x = torch.cat([z, action], dim=-1)
        return self.model(x)


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
        self.predictor = Predictor(embedding_dim=embedding_dim, action_dim=action_dim)
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

        preds = []
        if T_states > 1:
            for t in range(1, T_states):
                p = self.predictor(zs[:, t - 1], actions[:, t - 1])
                preds.append(p)
                # z_in = zs[:, t - 1] if self.training else zs[:, t]
                # a_in = actions[:, t - 1] if self.training else actions[:, t]
                # p = self.predictor(z_in, a_in)
                # preds.append(p)
        else:
            T_actions = actions.shape[1]
            # print(f"[JEPA.forward] Probing forward: iterative predictions with T_actions={T_actions}")
            z_prev = zs[:, 0]
            preds.append(z_prev)
            for t in range(T_actions):
                # print(f"[JEPA.forward] Iter {t}, z_prev.shape={z_prev.shape}, action.shape={actions[:, t].shape}")
                p = self.predictor(z_prev, actions[:, t])
                preds.append(p)
                z_prev = p

        if preds:
            preds = torch.stack(preds, dim=1)
        else:
            preds = torch.empty((B, 0, zs.size(-1)), device=zs.device)
        # print(f"[JEPA.forward] preds.shape={preds.shape}")

        with torch.no_grad():
            targ_z = self.target_encoder(flat).view(B, T_states, -1)[:, 1:]
        # print(f"[JEPA.forward] targ_z.shape={targ_z.shape}")

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
