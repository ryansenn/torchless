import torch
from transformers import AutoModelForCausalLM

MODEL_DIR = "../../Mistral-7B-v0.1"
ids = torch.tensor([[21558]], dtype=torch.long)

model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32).eval()

def make_hook(i):
    def hook(mod, inp, out):
        v = out[0, 0].detach().cpu().numpy()  # [hidden_size]
        print(f"layer {i} post_o_proj token0 first={v[0]} last={v[-1]}")
    return hook

for i, block in enumerate(model.model.layers):
    block.self_attn.o_proj.register_forward_hook(make_hook(i))

with torch.no_grad():
    _ = model(ids)