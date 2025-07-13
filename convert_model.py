import json
import sys
import os

if len(sys.argv) != 2:
    print("Wrong usage")
    sys.exit(1)

config_path = os.path.join(sys.argv[1], "config.json")

with open(config_path, "r") as f:
    config = json.load(f)

model = {
    "vocab_size": config["vocab_size"],
    "hidden_size": config["hidden_size"],
}


with open("model/model.json", "w") as f:
    json.dump(model, f, indent=4)