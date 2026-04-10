import torch
import torch.nn as nn

class BatchNorm(nn.Module):
    """Minimal BatchNorm for 2D inputs shaped as [batch_size, num_features]."""

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        if x.dim() != 2:
            raise ValueError(
                f"BatchNorm expects a 2D tensor of shape [batch_size, num_features], got {tuple(x.shape)}"
            )
        mean = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_normalized + self.beta
