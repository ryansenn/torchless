import json, os
from safetensors import safe_open

repo_dir = "./Mistral-7B-v0.1"
index = json.load(open(os.path.join(repo_dir, "model.safetensors.index.json")))
shard_file = index["weight_map"]["model.embed_tokens.weight"]   # e.g. "model-00001-of-00002.safetensors"
full_path = os.path.join(repo_dir, shard_file)

with safe_open(full_path, framework="pt", device="cpu") as f:
    w = f.get_tensor("model.embed_tokens.weight")
    print("shape:", w.shape)
    print("first weight:", w[0,0].item())