# torchless
I'm building a C++ inference engine from scratch that runs large language models directly from their raw weights. The first phase focuses on achieving a working inference pass for the [Mistral 7B model](https://huggingface.co/mistralai/Mistral-7B-v0.1), and the second phase will focus on performance improvements.

# Roadmap
- [x] **Model binary converter** *(scripts/export_mistral.py)*  
  Converts a Hugging Face Mistral model (config, vocab/merges, and weights) into a single standardized binary file that can be fed into the engine. It uses a JSON header to store model metadata, vocabulary, merges,     and tensor index information, followed by all model weights packed sequentially as contiguous floating point data.

- [x] **In-memory model loader**  *(src/loader/parameters.cpp)*  
  Memory-maps the binary, loads the config and provides direct tensor views without copying.

- [x] **Tokenizer**  *(src/tokenizer/tokenizer.cpp)*   
  The tokenizer implements a full byte-pair encoding (BPE) system compatible with Mistral’s vocabulary format. It loads the tokenizer.json, builds vocabulary and merge maps, applies Metaspace pre-tokenization           (replacing spaces with ▁), and encodes UTF-8 text by iteratively merging token pairs according to their rank. It also supports byte-fallback for unseen characters using the <0xNN> convention from the original         Mistral tokenizer.

- [x] **Tensor** *(src/common/tensor.cpp)*   
  Minimal tensor struct over float* with shape, size, strides, at(), reshape(), and copy helpers.

- [x] **CPU Ops** *(src/backend/cpu/kernels.cpp)*   
  Baseline implementations of core ops (e.g. matmul, softmax, RoPE) to be optimized later

- [ ] **Mistral architecture implementation** *(src/model/mistral/modules.cpp)*   
  Implementing each module with validation against PyTorch/HF
    - [x] Embedding
    - [x] RMSNorm
    - [x] Rotary Embedding
    - [ ] MLP
    - [ ] Attention
    - [ ] Decoder
    - [ ] LM Head
    - [ ] Model
    - [ ] KV Cache

- [ ] **CLI I/O**

# Resources

#### Concepts
- [Edward Z. Yang - PyTorch Internals](https://blog.ezyang.com/2019/05/pytorch-internals/)
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [Andrej Karpathy - Let’s build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
- [Positional Encoding Intuition](https://www.youtube.com/watch?v=T3OT8kqoqjc)
- [Rotary Embeddings](https://www.youtube.com/watch?v=V8r__fXx7tU)

#### Implementations
- [Andrew Chan - yalm](https://github.com/andrewkchan/yalm)
- [Hugging Face - Mistral model](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py)
- [Georgi Gerganov - GGML (tensor/operations)](https://github.com/ggml-org/llama.cpp/tree/master/ggml)

#### References
- [PyTorch Documentation](https://docs.pytorch.org/docs/stable/index.html)