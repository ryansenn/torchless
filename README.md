# torchless
I'm building a C++ inference engine from scratch that runs large language models directly from their raw weights. The first phase focuses on achieving a working inference pass for the [Mistral 7B model](https://huggingface.co/mistralai/Mistral-7B-v0.1), and the second phase will focus on performance improvements.

## Roadmap
- [x] **Model binary converter** *(scripts/export_mistral.py)*  
  Converts a Hugging Face Mistral model (config, vocab/merges, and weights) into a single standardized binary file that can be fed into the engine. It uses a JSON header to store model metadata, vocabulary, merges,     and tensor index information, followed by all model weights packed sequentially as contiguous floating point data.

- [x] **In-memory model loader**  *(src/loader/parameters.cpp)*  
  Memory-maps the binary and provides direct tensor views without copying.

- [x] **Tokenizer**  *(src/tokenizer/tokenizer.cpp)*   
  The tokenizer implements a full byte-pair encoding (BPE) system compatible with Mistral’s vocabulary format. It loads the tokenizer.json, builds vocabulary and merge maps, applies Metaspace pre-tokenization           (replacing spaces with ▁), and encodes UTF-8 text by iteratively merging token pairs according to their rank. It also supports byte-fallback for unseen characters using the <0xNN> convention from the original         Mistral tokenizer.

- [x] **Tensor** *(src/common/tensor.cpp)*   
  Minimal tensor struct over float* with shape, size, strides, at(), reshape(), and copy helpers.

- [x] **CPU Kernels** *(src/backend/cpu/kernels.cpp)*   
  Baseline implementations of core ops (e.g. matmul, softmax, pow) to be optimized later

- [ ] **Mistral architecture implementation** *(src/model/components.cpp)*
  - [ ] Implement each module and validate against its PyTorch reference
    - [ ] Embedding
    - [ ] RMSNorm
    - [ ] Rotary Embedding
    - [ ] MLP
    - [ ] Attention
    - [ ] Decoder
    - [ ] LM Head
    - [ ] Model
  - [ ] KV Cache

- [ ] **CLI I/O**

- [ ] **Performance**
  - [ ] Quantization
  - [ ] Parallelization
  - [ ] CUDA kernels

## Resources
- [Edward Z. Yang - PyTorch Internals](https://blog.ezyang.com/2019/05/pytorch-internals/)
- [Andrej Karpathy - Let’s build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
- [Hugging Face - Mistral Implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py)
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [PyTorch Documentation](https://docs.pytorch.org/docs/stable/index.html)