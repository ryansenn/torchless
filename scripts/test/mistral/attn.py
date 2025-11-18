import torch
from transformers import MistralConfig
from torch import nn
from typing import Optional, Callable

from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

from printer import show, dump, get_tensor
import safetensors

from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding

config = MistralConfig.from_json_file("../../../Mistral-7B-v0.1/config.json")

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    print(q_embed[0][0][1])
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

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
            attention_mask: Optional[torch.Tensor],
            past_key_values = None,
            cache_position = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        for i in range(1,4):
            dump("attn_q"+str(i), query_states.transpose(1, 2)[0][i-1])
            dump("attn_k"+str(i), key_states.transpose(1, 2)[0][i-1])
            dump("attn_v"+str(i), value_states.transpose(1, 2)[0][i-1])


        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


m = MistralAttention(config, 0)

state = {
    "q_proj.weight": get_tensor("model.layers.0.self_attn.q_proj.weight"),
    "k_proj.weight": get_tensor("model.layers.0.self_attn.k_proj.weight"),
    "v_proj.weight": get_tensor("model.layers.0.self_attn.v_proj.weight"),
    "o_proj.weight": get_tensor("model.layers.0.self_attn.o_proj.weight"),
}
m.load_state_dict(state)

emb = MistralRotaryEmbedding(config)
x = torch.tensor([(i % 128) / 10.0 for i in range(4 * 128)], dtype=torch.float32).view(1, 4, 1, 128)
position_ids = torch.tensor([[0,1,2]])
cos, sin = emb.forward(x,position_ids)

torch.manual_seed(0)
h0 = torch.randn(1, 1, 4096)
h1 = torch.randn(1, 1, 4096)
h2 = torch.randn(1, 1, 4096)

hidden_states = torch.cat([h0, h1, h2], dim=1)

config._attn_implementation = "eager"

mask_function = create_causal_mask if config.sliding_window is None else create_sliding_window_causal_mask
causal_mask = mask_function(
    config=config,
    input_embeds=hidden_states,
    attention_mask=None,
    cache_position=position_ids[0],
    past_key_values=None,
    position_ids=position_ids,
)

attn_output, attn_weights = m.forward(hidden_states,(cos,sin), causal_mask)

dump("attn_h1", h0)
dump("attn_h2", h1)
dump("attn_h3", h2)


#for i in range(1,4):
    #dump("attn_w" + str(i), attn_weights[0,:,i-1,:i])

dump("attn_o1", attn_output[0][0])
dump("attn_o2", attn_output[0][1])
dump("attn_o3", attn_output[0][2])




