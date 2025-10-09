# torchless
I'm building a C++ inference engine from scratch that runs large language models directly from their raw weights. The first phase focuses on achieving a working inference pass for the [Mistral 7B model](https://huggingface.co/mistralai/Mistral-7B-v0.1), and the second phase will focus on performance improvements.

## Roadmap
- [x] **Model binary converter** *(scripts/export_mistral.py)*  
  Converts a Hugging Face Mistral model (config, vocab/merges, and weights) into a single standardized binary file that can be fed into the engine. It uses a JSON header to store model metadata, vocabulary, merges,     and tensor index information, followed by all model weights packed sequentially as contiguous floating point data.

- [x] **In-memory model loader**  *(model/parameters.cpp)*  
  Memory-maps the binary and provides direct tensor views without copying.

- [x] **Tokenizer**  *(model/tokenizer.cpp)*   
  The tokenizer implements a full byte-pair encoding (BPE) system compatible with Mistral’s vocabulary format. It loads the tokenizer.json, builds vocabulary and merge maps, applies Metaspace pre-tokenization           (replacing spaces with ▁), and encodes UTF-8 text by iteratively merging token pairs according to their rank. It also supports byte-fallback for unseen characters using the <0xNN> convention from the original         Mistral tokenizer.

- [x] **Tensor** *(model/tensor.cpp)*   
  Minimal tensor struct over float* with shape, size, and strides; supports contiguous and strided views, at(), reshape(), and copy helpers for zero-copy access or in-place updates.

- [x] **Math Ops** *(inference/math_ops.cpp)*   
  Baseline implementation for matmul, normalization, activations, softmax, rotary embeddings, that will be optimized later

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
