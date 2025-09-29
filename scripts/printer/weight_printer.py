import json, os
from safetensors import safe_open
import torch

repo_dir = "../.././Mistral-7B-v0.1"

def get_tensor(tensor):
    index = json.load(open(os.path.join(repo_dir, "model.safetensors.index.json")))
    shard_file = index["weight_map"][tensor]   # e.g. "model-00001-of-00002.safetensors"
    full_path = os.path.join(repo_dir, shard_file)

    with safe_open(full_path, framework="pt", device="cpu") as f:
        w = f.get_tensor(tensor)

        return w
def print_first(tensor):
    w = get_tensor(tensor)
    print(tensor + " shape:", w.shape)
    print("first weight:", w.flatten()[0].item())

'''
print_first("model.embed_tokens.weight")
print_first("model.layers.0.input_layernorm.weight")
print_first("model.layers.14.input_layernorm.weight")
print_first("model.layers.31.input_layernorm.weight")
'''

'''
print_first("model.layers.0.self_attn.q_proj.weight")
print_first("model.layers.0.self_attn.k_proj.weight")
print_first("model.layers.0.self_attn.v_proj.weight")
'''

#print_first("model.layers.31.self_attn.q_proj.weight")
#print_first("model.layers.31.self_attn.k_proj.weight")
#print_first("model.layers.31.self_attn.v_proj.weight")

'''
torch.set_printoptions(precision=8, sci_mode=False)

def f(i):
    w_k = get_tensor("model.layers." + str(i) + ".self_attn.k_proj.weight").to(torch.float32)
    x = torch.ones(4096, dtype=torch.float32)
    y = w_k @ x
    print(y.shape, y.dtype)
    print((y[0], y[1023]))

f(0)
f(31)
'''

print_first("model.layers.0.self_attn.o_proj.weight")
print_first("model.layers.31.self_attn.o_proj.weight")

