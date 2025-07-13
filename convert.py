import json
import sys
import os
import struct

if len(sys.argv) != 2:
    print("Wrong usage")
    sys.exit(1)

config_path = os.path.join(sys.argv[1], "config.json")

with open(config_path, "r") as f:
    config = json.load(f)


def add_metadata_entry(model, key, value):

    model.append(struct.pack("B", 0)) # entry type
    model.append(struct.pack("i", len(key))) # key size
    model.append(key.encode("utf-8")) # key

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


# key length, key, value type, value lenght, value
def add_tensor_entry(model, key, value):
    model.append(struct.pack("B", 1))

model = []

add_metadata_entry(model, "vocab_size", config["vocab_size"])
add_metadata_entry(model, "hidden_size", config["hidden_size"])

with open("model.bin", "wb") as f:
    for entry in model:
        f.write(entry)