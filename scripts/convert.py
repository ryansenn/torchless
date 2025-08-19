import json
import sys
import os
import struct
import safetensors
from safetensors.torch import safe_open
import torch

"""
Builds a binary model file "model.bin" from the Mistral-7B-v0.1 directory.

Reads:
    - config.json (model config)
    - tokenizer.json (vocabulary)
    - model.safetensors.index.json and referenced .safetensors files (tensor weights)

Binary format:
    Keys are fixed-width (50 bytes, zero-padded).
    Entry types:
        Metadata entry (0): key, value type (int=0, float=1, string=2), value
        Tensor entry (1): key, dtype code (bfloat16=0, float16=1, float32=2, uint8=3), byte size, raw tensor data
"""

MODEL_PATH = "Mistral-7B-v0.1/"

if len(sys.argv) > 2:
    MODEL_PATH = sys.argv[1]

model = []

config_path = os.path.join(MODEL_PATH, "config.json")
tensor_index_path = os.path.join(MODEL_PATH, "model.safetensors.index.json")
tokenizer_path = os.path.join(MODEL_PATH, "tokenizer.json")

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
    

# key, value type, value length, tensor
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
    tensor_path = os.path.join(MODEL_PATH, tensor_index["weight_map"][key])
                               
    with safetensors.safe_open(tensor_path, framework="pt") as f:
        return f.get_tensor(key)

def load_vocab():
    words = [b"" for i in range(config["vocab_size"])]
    with open(tokenizer_path, "r") as f:
        vocab = json.load(f)["model"]["vocab"]

        for k in vocab:
            word = k
            word = word.replace('\u2581', ' ') # check if this is really needed with mistral
            word = word.replace("\0", "\7") # check if this is really needed with mistral
            word = word+'\0'
            words[vocab[k]] = word.encode("utf-8")


    buf = bytearray(b"".join(words))
    tensor = torch.frombuffer(buf, dtype=torch.uint8)

    add_tensor_entry(model, "vocab", tensor)
    

add_metadata_entry(model, "vocab_size", config["vocab_size"])
add_metadata_entry(model, "hidden_size", config["hidden_size"])

load_vocab()

add_tensor_entry(model, "model.embed_tokens.weight", get_tensor("model.embed_tokens.weight").to(torch.float32))

out_path = "model.bin"
with open(out_path, "wb") as f:
    for entry in model:
        f.write(entry)

size = os.path.getsize(out_path)
print(f"OK: wrote {out_path} ({size} bytes)")
