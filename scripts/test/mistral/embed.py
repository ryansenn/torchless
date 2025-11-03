import torch
import safetensors

f = "../../../Mistral-7B-v0.1/model-00001-of-00002.safetensors"

emb = torch.nn.Embedding(32000, 4096)

with safetensors.safe_open(f, framework="pt") as f:
    tensor = f.get_tensor("model.embed_tokens.weight")
    print(tensor[0][0])
    print(tensor[0][-1])
    print(tensor[-1][0])
    print(tensor[-1][-1])