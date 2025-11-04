import torch

def show(tensor):
    for v in tensor.flatten():
        print(v.item(), end=" ")

    print("")