import torch
from transformers import MistralConfig
from torch import nn
from typing import Optional
from printer import show, dump

from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb, MistralRotaryEmbedding

config = MistralConfig.from_json_file("../../../Mistral-7B-v0.1/config.json")

class MistralAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MistralConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        return query_states, key_states, value_states


m = MistralAttention(config, 0)
emb = MistralRotaryEmbedding(config)
x = torch.tensor([(i % 128) / 10.0 for i in range(4 * 128)], dtype=torch.float32).view(1, 4, 1, 128)
position_ids = torch.tensor([[0]])
cos, sin = emb.forward(x,position_ids)

h = torch.randn(1,1,4096)

q,k,v = m.forward(h,(cos,sin))

dump("attn1_h", h)
dump("attn1_q", q)
dump("attn1_k", k)
dump("attn1_v", v)
