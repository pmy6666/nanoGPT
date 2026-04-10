import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """Minimal LayerNorm over the last dimension."""

    def __init__(self, n_embd, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(n_embd))
        self.beta = nn.Parameter(torch.zeros(n_embd))

    def forward(self, x):
        if x.size(-1) != self.gamma.numel():
            raise ValueError(
                f"LayerNorm expects last dim {self.gamma.numel()}, got {x.size(-1)}"
            )
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_normalized + self.beta
