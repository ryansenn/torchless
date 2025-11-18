from typing import Optional

import torch
from transformers import MistralConfig
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralRotaryEmbedding

from printer import get_tensor, dump

config = MistralConfig.from_json_file("../../../Mistral-7B-v0.1/config.json")

layer = MistralDecoderLayer(config,0)

state = {
    "self_attn.q_proj.weight": get_tensor("model.layers.0.self_attn.q_proj.weight"),
    "self_attn.k_proj.weight": get_tensor("model.layers.0.self_attn.k_proj.weight"),
    "self_attn.v_proj.weight": get_tensor("model.layers.0.self_attn.v_proj.weight"),
    "self_attn.o_proj.weight": get_tensor("model.layers.0.self_attn.o_proj.weight"),

    "mlp.gate_proj.weight":    get_tensor("model.layers.0.mlp.gate_proj.weight"),
    "mlp.up_proj.weight":      get_tensor("model.layers.0.mlp.up_proj.weight"),
    "mlp.down_proj.weight":    get_tensor("model.layers.0.mlp.down_proj.weight"),

    "input_layernorm.weight":  get_tensor("model.layers.0.input_layernorm.weight"),
    "post_attention_layernorm.weight": get_tensor("model.layers.0.post_attention_layernorm.weight"),
}

layer.load_state_dict(state)

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

y = layer.forward(hidden_states, attention_mask=causal_mask, position_ids=position_ids, position_embeddings=(cos,sin))

print(y.shape)

dump("layer_h1", h0)
dump("layer_h2", h1)
dump("layer_h3", h2)

dump("layer_o1", y[0][0])
dump("layer_o2", y[0][1])
dump("layer_o3", y[0][2])