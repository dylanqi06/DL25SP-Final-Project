from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class MockModel(torch.nn.Module):
    """
    Does nothing. Just for testing.
    """

    def __init__(self, device="cuda", output_dim=256):
        super().__init__()
        self.device = device
        self.repr_dim = output_dim

    def forward(self, states, actions):
        """
        Args:
            During training:
                states: [B, T, Ch, H, W]
            During inference:
                states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        B, T, _ = actions.shape

        return torch.randn((B, T + 1, self.repr_dim)).to(self.device)

class Encoder(nn.Module):
    def __init__(self, input_shape=(3, 64, 64), embedding_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # (64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # (128, 8, 8)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, embedding_dim),
        )

    def forward(self, x):  # x: [B, 3, 64, 64]
        return self.cnn(x)

class Predictor(nn.Module):
    def __init__(self, embedding_dim=256, action_dim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, z, action):
        x = torch.cat([z, action], dim=-1)
        return self.model(x)

class JEPA(nn.Module):
    def __init__(self, embedding_dim=256, action_dim=2):
        super().__init__()
        self.encoder = Encoder(embedding_dim=embedding_dim)
        self.predictor = Predictor(embedding_dim=embedding_dim, action_dim=action_dim)

    def forward(self, observations, actions):
        """
        observations: [B, T, 3, 64, 64]
        actions:      [B, T-1, 2]
        Returns: predicted embeddings [B, T-1, D], target embeddings [B, T-1, D]
        """
        B, T, C, H, W = observations.shape
        z = self.encoder(observations.view(-1, C, H, W))     # [B*T, D]
        z = z.view(B, T, -1)                                 # [B, T, D]

        pred_embeddings = []
        for t in range(1, T):
            z_prev = z[:, t - 1]      # [B, D]
            a_prev = actions[:, t - 1]  # [B, 2]
            z_pred = self.predictor(z_prev, a_prev)  # [B, D]
            pred_embeddings.append(z_pred)

        pred_embeddings = torch.stack(pred_embeddings, dim=1)  # [B, T-1, D]
        target_embeddings = z[:, 1:]  # [B, T-1, D]

        return pred_embeddings, target_embeddings


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output
