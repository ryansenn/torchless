import torch
from torch import nn
from transformers import MistralConfig
from transformers.activations import ACT2FN
from printer import dump, get_tensor


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
        #dump("first", self.gate_proj(x)[:5])
        dump("second", self.up_proj(x)[:5])
        #dump("third", self.act_fn(self.gate_proj(x))[:5])
        #dump("fourth", (self.act_fn(self.gate_proj(x)) * self.up_proj(x))[:5])
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

config = MistralConfig.from_json_file("../../../Mistral-7B-v0.1/config.json")

torch.manual_seed(0)
h = torch.randn(4096)

mlp = MistralMLP(config)

state = {
    "down_proj.weight": get_tensor("model.layers.0.mlp.down_proj.weight"),
    "gate_proj.weight": get_tensor("model.layers.0.mlp.gate_proj.weight"),
    "up_proj.weight": get_tensor("model.layers.0.mlp.up_proj.weight"),
}
mlp.load_state_dict(state)

y = mlp.forward(h)

#dump("mlp_h", h)
#dump("mlp_output", y)

