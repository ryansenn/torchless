import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "hello how are you"

# Encode to token IDs
encoded = tokenizer(text, return_tensors="pt")
print(encoded["input_ids"])
