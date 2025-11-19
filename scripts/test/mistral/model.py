from transformers import MistralModel
import torch

model = MistralModel.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    torch_dtype=torch.float32,
    device_map="auto"
)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# The text to process
text = ""

# 2. Prepare the Input Tensor
input_ids = tokenizer.encode(text, return_tensors="pt")

print(input_ids)

input_ids = input_ids.to(model.device)

with torch.no_grad():
    outputs = model(input_ids)

last_hidden_state = outputs.last_hidden_state

print("Forward pass complete. Output shape:", last_hidden_state.shape)