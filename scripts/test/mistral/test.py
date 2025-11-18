from transformers import AutoTokenizer, MistralForCausalLM

model = MistralForCausalLM.from_pretrained("meta-mistral/Mistral-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-mistral/Mistral-2-7b-hf")

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")


generate_ids = model.generate(inputs.input_ids, max_length=30)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."