import torch

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