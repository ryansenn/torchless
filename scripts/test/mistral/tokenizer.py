import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "The quick brown fox jumps over the lazy dog. Mistral models tokenize text using byte pair encoding, handling punctuation, emojis 😊, and multilingual words like déjà vu or 中文 with care."
text2 = "A\uE000B"

# Encode to token IDs
encoded = tokenizer(text2, return_tensors="pt")
#print(encoded["input_ids"])

for i in encoded["input_ids"]:
    for j in i:
        print(j.item())
