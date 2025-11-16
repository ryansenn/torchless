# torchless
Torchless is an LLM inference engine built from scratch

# Roadmap

- [x] **Model Loader**
  - [x] **Binary converter** *(scripts/export_mistral.py)*  
    Converts a Hugging Face Mistral model (config, vocab/merges, and weights) into a single standardized binary file that can be fed into the engine. It uses a JSON header to store model metadata, vocabulary, merges, and tensor index information, followed by all model weights packed sequentially as contiguous floating point data.
  - [x] **In-memory loader** *(src/loader/parameters.cpp)*  
    Memory-maps the binary, loads the config and provides direct tensor views.

- [x] **Tensor & Ops**
  - [x] Tensor implemented as a view over memory with shape/strides *(src/common/tensor.cpp)*
  - [x] Math operations (e.g. matmul, softmax, RoPE) to be optimized later *(src/backend/cpu/kernels.cpp)*

- [x] **Tokenizer**  *(src/tokenizer/tokenizer.cpp)*     
  The tokenizer implements full byte-pair encoding (BPE) compatible with Mistral’s vocabulary. It loads tokenizer.json, builds vocab and merge maps, applies Metaspace pre-tokenization, encodes UTF-8 text by merging token pairs by rank, and supports byte fallback

- [x] **Inference State** *(src/common/inference_state.h)*  
  Holds temporary memory and KV cache used during inference

- [ ] **Mistral architecture implementation** *(src/model/mistral/modules.cpp)*
  - Each module implemented and tested against PyTorch/HF
    - [x] **Embedding** - Looks up initial embedding from token ids.
    - [x] **RMSNorm** - computes the RMS over the current hidden_state, normalizes it, and applies the learned gain vector g
    - [x] **Rotary Embedding** - precomputes inverse frequencies from rope_theta and fills cos/sin tensors for RoPE for each position.
    - [x] **Attention** - projects hidden_state into Q/K/V, applies RoPE to Q and K, updates the KV cache, runs grouped-query attention over the window, then applies the output projection back into hidden_state
    - [x] **Feedforward MLP** - implements the SwiGLU feedforward: linear projections + SiLU
    - [x] **Layer** - runs norm, attention, and MLP with residuals around each subblock
    - [ ] **LM Head**

- [ ] **CLI I/O**

- [ ] **Quantization**

- [ ] **Parallelization**

- [ ] **Custom CUDA Kernels**

# Resources

#### Concepts
- [Edward Z. Yang - PyTorch Internals](https://blog.ezyang.com/2019/05/pytorch-internals/)
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [Andrej Karpathy - Let’s build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
- [Positional Encoding Intuition](https://www.youtube.com/watch?v=T3OT8kqoqjc)
- [Rotary Embeddings](https://www.youtube.com/watch?v=V8r__fXx7tU)

#### Implementations
- [Hugging Face - Mistral model](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py)
- [Andrew Chan - yalm](https://andrewkchan.dev/posts/yalm.html)
- [Georgi Gerganov - GGML (tensor/operations)](https://github.com/ggml-org/llama.cpp/tree/master/ggml)

My C++ Mistral architecture implementation matches the HF transformers python implementation. Each module has been checked against it.

Andrew Chan's yaml project was the inspiration for starting this project, strongly recommend his blog posts.