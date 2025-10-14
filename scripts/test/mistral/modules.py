import torch
from torch import nn

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


x = torch.tensor([2.4, 7.8, 1.1, 5.3, 9.0, 4.7, 6.2, 3.8, 8.5, 0.6])
g = torch.tensor([0.8, 1.2, 1.0, 0.9, 1.1, 1.3, 0.7, 1.0, 1.2, 0.95])
norm = MistralRMSNorm(g)
y = norm(x)
print(y)

