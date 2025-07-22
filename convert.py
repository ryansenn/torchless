import json
import sys
import os
import struct
import safetensors
from safetensors.torch import safe_open
import torch


if len(sys.argv) != 2:
    print("Wrong usage")
    sys.exit(1)

model = []

config_path = os.path.join(sys.argv[1], "config.json")
tensor_index_path = os.path.join(sys.argv[1], "model.safetensors.index.json")
tokenizer_path = os.path.join(sys.argv[1], "tokenizer.json")

with open(config_path, "r") as f:
    config = json.load(f)

with open(tensor_index_path, "r") as f:
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

    if tensor.dtype == torch.bfloat16:
        model.append(struct.pack("B", 0))
    elif tensor.dtype == torch.float16:
        model.append(struct.pack("B", 1))
    elif tensor.dtype == torch.float32:
        model.append(struct.pack("B", 2))
    elif tensor.dtype == torch.uint8:
        model.append(struct.pack("B", 3))

    size = tensor.element_size() * tensor.numel()
    model.append(struct.pack("q", size))
    model.append(tensor.numpy().tobytes())
    
        
def get_tensor(key):
    tensor_path = os.path.join(sys.argv[1], tensor_index["weight_map"][key])
                               
    with safetensors.safe_open(tensor_path, framework="pt") as f:
        return f.get_tensor(key)

def load_vocab():
    words = ["" for i in range(config["vocab_size"])]
    with open(tokenizer_path, "r") as f:
        vocab = json.load(f)["model"]["vocab"]

        for k in vocab:
            word = k
            word = word.replace('\u2581', ' ') # check if this is really needed with mistral
            word = word.replace("\0", "\7") # check if this is really needed with mistral
            word = word+'\0'
            words[vocab[k]] = word.encode("utf-8")

    tensor = torch.cat([torch.tensor(w) for w in words])

    add_tensor_entry(model, "vocab", tensor)
    

add_metadata_entry(model, "vocab_size", config["vocab_size"])
add_metadata_entry(model, "hidden_size", config["hidden_size"])

load_vocab()

add_tensor_entry(model, "model.embed_tokens.weight", get_tensor("model.embed_tokens.weight").to(torch.float32))

with open("model.bin", "wb") as f:
    for entry in model:
        f.write(entry)