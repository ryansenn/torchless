import torch
from torch import nn
from transformers import MistralConfig
from transformers.activations import ACT2FN
from printer import dump


class MistralMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

config = MistralConfig.from_json_file("../../../Mistral-7B-v0.1/config.json")

torch.manual_seed(0)
h = torch.randn(4096)

mlp = MistralMLP(config)

y = mlp.forward(h)

dump("mlp_h", h)
dump("mlp_output", y)



