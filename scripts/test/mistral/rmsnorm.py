import torch
from torch import nn
from printer import show, dump

class MistralRMSNorm(nn.Module):
    def __init__(self, g, eps=1e-6):
        """
        MistralRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(g)
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


x = torch.randn(4096)
g = torch.randn(4096)

norm = MistralRMSNorm(g)
y = norm(x)

dump("norm_x",x)
dump("norm_g", g)
dump("norm_y", y)


