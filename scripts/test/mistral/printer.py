import torch
import safetensors

f1 = "../../../Mistral-7B-v0.1/model-00001-of-00002.safetensors"
f2 = "../../../Mistral-7B-v0.1/model-00002-of-00002.safetensors"

def get_tensor(name):
    with safetensors.safe_open(f1, framework="pt") as o1:
        try:
            return o1.get_tensor(name)
        except:
            pass
    with safetensors.safe_open(f2, framework="pt") as o2:
        return o2.get_tensor(name)
def dump(name, tensor):
    with open("../output.txt", "a") as f:
        f.write(name + "\n")

        for v in tensor.flatten():
            f.write(str(v.item()) + " ")

        f.write("\n\n")

def show(tensor):
    for v in tensor.flatten():
        print(v.item(), end=" ")

    print("")