import torch
from transformers import MistralConfig
from printer import get_tensor, dump

config = MistralConfig.from_json_file("../../../Mistral-7B-v0.1/config.json")

lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
lm_head.load_state_dict({"weight": get_tensor("lm_head.weight")})

torch.manual_seed(0)
x = torch.rand(4096)

y = lm_head.forward(x)


dump("lmhead_x", x)

#dump("lmhead_y", y)

print(y[0])
print(y[31999])