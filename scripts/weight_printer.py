import json, os
from safetensors import safe_open

repo_dir = ".././Mistral-7B-v0.1"

def print_first(tensor):
    index = json.load(open(os.path.join(repo_dir, "model.safetensors.index.json")))
    shard_file = index["weight_map"][tensor]   # e.g. "model-00001-of-00002.safetensors"
    full_path = os.path.join(repo_dir, shard_file)

    with safe_open(full_path, framework="pt", device="cpu") as f:
        w = f.get_tensor(tensor)
    print(tensor + " shape:", w.shape)
    print("first weight:", w.flatten()[0].item())

'''
print_first("model.embed_tokens.weight")
print_first("model.layers.0.input_layernorm.weight")
print_first("model.layers.14.input_layernorm.weight")
print_first("model.layers.31.input_layernorm.weight")
'''

print_first("model.layers.0.self_attn.q_proj.weight")
print_first("model.layers.0.self_attn.k_proj.weight")
print_first("model.layers.0.self_attn.v_proj.weight")

#print_first("model.layers.31.self_attn.q_proj.weight")
#print_first("model.layers.31.self_attn.k_proj.weight")
#print_first("model.layers.31.self_attn.v_proj.weight")