# torchless

A C++ inference engine that runs LLMs directly from model weights.

I'm currently focusing on getting a first working CPU inference pass for the [Mistral-7B model](https://huggingface.co/mistralai/Mistral-7B-v0.1)

## Todo
- [x] model binary converter
- [x] model in-memory loader
- [x] tokenizer
- [x] tensor + math ops
- [ ] transformer forward pass
    - [x] embeddings
    - [ ] attention
    - [ ] mlp
    - [ ] output
- [ ] fix segfaults
- [ ] cli input/output
- [ ] optimize speed
    - [ ] parallelization
    - [ ] quantization
    - [ ] CUDA kernel?  

Resources: 

- [Attention is all you need](https://arxiv.org/pdf/1706.03762)
- [Hugging face mistral implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py)
