import os
import json
import struct

import safetensors
import torch

"""
This script converts a Hugging Face Mistral model into one standardized binary file that can be fed into the inference engine.

Inputs (from the downloaded model directory):
  - config.json              : model hyperparameters
  - tokenizer.json           : vocabulary
  - model.safetensors.index.json + shard files : weights

Output:
  model.bin with layout:
    [8-byte uint64: size of JSON header]
    [JSON header: config, vocab, tensor index]
    [payload: All tensors as float32]
    
How it works:
- Read config.json (metadata), tokenizer.json (vocab), and safetensors index (tensors).
- Build a JSON header with metadata, vocab, and tensor offsets.
- Write to model.bin:
    1) 8-byte header size (uint64, little-endian)
    2) JSON header (UTF-8)
    3) All tensors as contiguous float32
"""

IN_PATH = "../Mistral-7B-v0.1"
OUT_PATH = "../model.bin"

header = {}

# Load config inside of header["config"]
config_path = os.path.join(IN_PATH, "config.json")

with open(config_path, 'r') as f:
    cfg = json.load(f)
    header["metadata"] = {
        "hidden_size": str(cfg["hidden_size"]),
        "intermediate_size": str(cfg["intermediate_size"]),
        "n_layers": str(cfg["num_hidden_layers"]),
        "n_heads": str(cfg["num_attention_heads"]),
        "n_kv_heads": str(cfg["num_key_value_heads"]),
        "vocab_size": str(cfg["vocab_size"]),
        "max_seq_len": str(cfg["max_position_embeddings"]),
        "rope_theta": str(cfg["rope_theta"]),
        "norm_eps": str(cfg["rms_norm_eps"]),
        "act_type": cfg["hidden_act"],
        "dtype": "fp32",
    }

# Insert the vocab in header["vocab"]
tokenizer_path = os.path.join(IN_PATH, "tokenizer.json")

with open(tokenizer_path, 'r') as f:
    header["vocab"] = json.load(f)["model"]["vocab"]


# Load weight map
tensor_index_path = os.path.join(IN_PATH, "model.safetensors.index.json")
with open(tensor_index_path, 'r') as f:
    index = json.load(f)
    weight_map = index["weight_map"]


# Loop through each tensor and add info to header["tensors"]
header["tensors"] = {}
raw_tensors = {}
start = 0

for tensor_name in weight_map:
    tensor_file_path = os.path.join(IN_PATH, weight_map[tensor_name])

    with safetensors.safe_open(tensor_file_path, framework="pt") as f:
        tensor = f.get_tensor(tensor_name)
        header["tensors"][tensor_name] = {"dtype": "F32", "shape": list(tensor.shape)[:4], "offset": start}
        start += tensor.numel() * 4 # Here we change when we quantize


# Serialize header as UTF-8 bytes
header_bytes = json.dumps(header).encode("utf-8")

# Get header size as an 8-byte little-endian unsigned integer
header_size = struct.pack("<Q", len(header_bytes))


### Write out everyting to binary output file
with open(OUT_PATH, "wb") as out:
    out.write(header_size)
    out.write(header_bytes)

    # Dump all the tensors in the same order as header
    for tensor_name in weight_map:
        tensor_file_path = os.path.join(IN_PATH, weight_map[tensor_name])

        with safetensors.safe_open(tensor_file_path, framework="pt") as f:
            tensor = f.get_tensor(tensor_name)

            # Convert to F32 tensor (change here when we quantize)
            tensor = tensor.to(torch.float32).contiguous()

            # Turn tensor into bytes
            tensor_bytes = tensor.numpy().tobytes()

            out.write(tensor_bytes)
