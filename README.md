# torchless
C++ inference engine that runs LLMs directly from model weights

## Roadmap
- [x] **Model binary converter** *(scripts/export_mistral.py)*  
  Converts a Hugging Face Mistral model (config, vocab, and weights) into a single binary file.
- [x] **In-memory model loader**  
  Memory-maps the binary and provides direct tensor views without copying.
- [ ] **Tokenizer** *(rewriting)*  
  Replacing the simple trie-based tokenizer with a full byte-level BPE implementation that matches Mistral’s Hugging Face tokenizer (Metaspace pre-tokenization + byte fallback).
- [x] **Tensor + math ops**  
  Tensor struct for shape, strides, and memory views, + math ops (matmul, normalization, activations, softmax, rotary embeddings)
- [ ] **Transformer forward pass**  
  - [ ] Attention *(being rewritten)*  
  - [ ] MLP  
  - [ ] Output layer
- [ ] **Fix segfaults**
- [ ] **CLI I/O**
- [ ] **Performance**  
  - [ ] Parallelization  
  - [ ] Quantization  
  - [ ] CUDA kernels

## Resources
- [Karpathy - Let’s build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [Mistral implementation (Hugging Face)](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py)
