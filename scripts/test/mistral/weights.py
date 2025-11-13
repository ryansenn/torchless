import safetensors
import torch

f1 = "../../../Mistral-7B-v0.1/model-00001-of-00002.safetensors"
f2 = "../../../Mistral-7B-v0.1/model-00002-of-00002.safetensors"


def list_tensors(f):
    with safetensors.safe_open(f, framework="pt") as f:
        for name in f.keys():
            tensor = f.get_tensor(name)
            flat = tensor.flatten()
            print(name, tensor.shape, flat[0].item(), flat[-1].item())



list_tensors(f1)
list_tensors(f2)