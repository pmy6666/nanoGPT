import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Minimal RMSNorm over the last dimension."""

    def __init__(self, n_embd, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(n_embd))

    def forward(self, x):
        if x.size(-1) != self.gamma.numel():
            raise ValueError(
                f"RMSNorm expects last dim {self.gamma.numel()}, got {x.size(-1)}"
            )
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_normalized = x / rms
        return self.gamma * x_normalized
