# torchless
C++ inference engine that runs LLMs directly from model weights

## Roadmap
- [x] Model binary converter
- [x] In-memory model loader
- [ ] Tokenizer *(rewriting)*
- [x] Tensor + math ops
- [ ] Transformer forward pass  
  - [ ] Attention *(rewriting)*  
  - [ ] MLP  
  - [ ] Output
- [ ] Fix segfaults
- [ ] CLI I/O
- [ ] Performance  
  - [ ] Parallelization  
  - [ ] Quantization  
  - [ ] CUDA kernels

## Resources
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [Transformers: Mistral implementation (HF)](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py)
