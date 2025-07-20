import json
import sys
import os
import struct
from safetensors.torch import safe_open
import torch


if len(sys.argv) != 2:
    print("Wrong usage")
    sys.exit(1)

config_path = os.path.join(sys.argv[1], "config.json")
tensor_index = os.path.join(sys.argv[1], "model.safetensors.index.json")

with open(config_path, "r") as f:
    config = json.load(f)

with open(tensor_index, "r") as f:
    tensor_index = json.load(f)


def add_metadata_entry(model, key, value):

    model.append(struct.pack("B", 0)) # entry type
    model.append(key.encode("utf-8").ljust(50, b'\0')[:50]) # key size is always 50

    if isinstance(value, int):
        model.append(struct.pack("B", 0))  # Value type: int
        model.append(struct.pack("i", value))
    elif isinstance(value, float):
        model.append(struct.pack("B", 1))  # Value type: float
        model.append(struct.pack("f", value))
    elif isinstance(value, str): 
        val_bytes = value.encode("utf-8")
        model.append(struct.pack("B", 2))  # Value type: string
        model.append(struct.pack("I", len(val_bytes)))
        model.append(val_bytes)
    

# key length, key, value type, value lenght, tensor
def add_tensor_entry(model, key, tensor):
    model.append(struct.pack("B", 1))
    model.append(key.encode("utf-8").ljust(50, b'\0')[:50]) # key size is always 50
    tensor = tensor.to(torch.float32) # always use f32 for now 

    if tensor.dtype == torch.bfloat16:
        model.append(struct.pack("B", 0))
    elif tensor.dtype == torch.float16:
        model.append(struct.pack("B", 1))
    elif tensor.dtype == torch.float32:
        model.append(struct.pack("B", 2))

    size = tensor.element_size() * tensor.numel()
    model.append(struct.pack("q", size))
    model.append(tensor.numpy().tobytes())
    
        
def get_tensor(key):
    tensor_path = os.path.join(sys.argv[1], tensor_index["weight_map"][key])
                               
    with safetensors.safe_open(tensor_path, framework="pt") as f:
        return f.get_tensor(key)
    

model = []

add_metadata_entry(model, "vocab_size", config["vocab_size"])
add_metadata_entry(model, "hidden_size", config["hidden_size"])

add_tensor_entry(model, "model.embed_tokens.weight", get_tensor("model.embed_tokens.weight"))

with open("model.bin", "wb") as f:
    for entry in model:
        f.write(entry)