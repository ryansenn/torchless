import os
import json
import struct

import safetensors
import torch

import argparse

from quantize import quantize

"""
Usage:
  python export_mistral.py --model_dir /path/to/Mistral-7B-v0.1 [--out model.bin] [--quant int8]

Arguments:
  --model_dir   Required. Path to the Hugging Face model directory.
  --out         Optional. Output file path. Defaults to ./model.bin
  --quant       Optional. Quantization mode. Defaults to f32. Accepts f32 or int8
"""

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", required=True)
parser.add_argument("--out", default="./model.bin")
parser.add_argument("--quant", default="f32", choices=["f32", "int8"])
args = parser.parse_args()

IN_PATH = args.model_dir
OUT_PATH = args.out
DATA_SIZE = 4
DATA_TYPE = torch.float32
GROUP_SIZE = 64

if args.quant == "int8":
    DATA_SIZE = 1
    DATA_TYPE = torch.int8

"""
This script converts a Hugging Face Mistral model into one standardized binary file that can be fed into the inference engine.

Inputs (from the downloaded model directory):
  - config.json              : model hyperparameters
  - tokenizer.json           : vocabulary
  - model.safetensors.index.json + shard files : weights

Output:
  model.bin with layout:
    [8-byte uint64: size of JSON header]
    [JSON header]: 
        - config
        - vocab/merges
        - tensor info:
            - data type
            - shape
            - tensor start index
            - scales size
            - scales start index
            
    [payload: All tensors as continuous data with quantization scales]
"""

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
        "max_position_embeddings": str(cfg["max_position_embeddings"]),
        "sliding_window": str(cfg["sliding_window"]),
        "rope_theta": str(cfg["rope_theta"]),
        "norm_eps": str(cfg["rms_norm_eps"]),
        "act_type": cfg["hidden_act"]
    }

# Insert the vocab in header["vocab"]
tokenizer_path = os.path.join(IN_PATH, "tokenizer.json")
header["tokenizer"] = {}

with open(tokenizer_path, 'r') as f:
    t = json.load(f)
    header["tokenizer"]["vocab"] = t["model"]["vocab"]
    header["tokenizer"]["merges"] = t["model"]["merges"]


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
        header["tensors"][tensor_name] = {"dtype": args.quant, "shape": list(tensor.shape)[:4], "offset": start}

        start += tensor.numel() * DATA_SIZE

        if args.quant != "f32":
            # We store scales
            scale_size = tensor.numel() // GROUP_SIZE * 4
            header["tensors"][tensor_name]["scale_start"] = start
            header["tensors"][tensor_name]["scale_size"] = scale_size
            start += tensor.numel() // GROUP_SIZE * 4


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
            scales = None

            if args.quant != "f32":
                tensor, scales = quantize(tensor, DATA_SIZE * 8, GROUP_SIZE)
                scales = scales.to(torch.float32).contiguous()

            tensor = tensor.to(DATA_TYPE).contiguous()

            # Turn tensor into bytes and write
            tensor_bytes = tensor.numpy().tobytes()
            out.write(tensor_bytes)

            # Write scales
            if scales is not None:
                scales_bytes = scales.numpy().tobytes()
                out.write(scales_bytes)
