import torch.nn as nn


class TinyMLP(nn.Module):
    """Small MLP surrogate model for predicting ODE states."""

    def __init__(self, input_dim, output_dim=6, hidden_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)
