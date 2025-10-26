from transformers import MistralConfig
from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding
import torch

config = MistralConfig.from_json_file("../../../Mistral-7B-v0.1/config.json")

emb = MistralRotaryEmbedding(config)

#print(" ".join(str(i.item()) for i in emb.inv_freq))

x = torch.tensor([])
position_ids = torch.tensor([[0,3]])


cos, sin = emb.forward(x,position_ids)

cos = cos.squeeze(0)
sin = sin.squeeze(0)

print(cos[0][:len(cos[0])//2])
print(cos[1][:len(cos[1])//2])

print("")

print(sin[0][:len(sin[0])//2])
print(sin[1][:len(sin[1])//2])


