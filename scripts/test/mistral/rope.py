from transformers import MistralConfig
from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding
import torch

config = MistralConfig.from_json_file("../../../Mistral-7B-v0.1/config.json")

emb = MistralRotaryEmbedding(config)
x = torch.tensor([(i % 128) / 10.0 for i in range(4 * 128)], dtype=torch.float32).view(1, 4, 1, 128)
position_ids = torch.tensor([[0,1,2,3]])
cos, sin = emb.forward(x,position_ids)

"""
cos = cos.squeeze(0)
sin = sin.squeeze(0)
print(cos[0][:len(cos[0])//2])
print(cos[1][:len(cos[1])//2])
print("")
print(sin[0][:len(sin[0])//2])
print(sin[1][:len(sin[1])//2])
"""

q = torch.tensor([(i % 128) / 256 for i in range(4 * 128)])
q = q.view(1,1, 4, 128)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

q2, k = apply_rotary_pos_emb(q, q, cos, sin)
print(q2[0][0][1])

