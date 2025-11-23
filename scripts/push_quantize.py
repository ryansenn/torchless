from huggingface_hub import login
login()

import torch
from transformers import MistralModel, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
)

model = MistralModel.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
)

model.push_to_hub("ryansen/Mistral-7B-v0.1-8bit")