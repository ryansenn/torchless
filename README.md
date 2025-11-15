# torchless
I'm building an inference engine from scratch that runs large language models directly from their raw weights. This project is educational meant to learn LLM inference at a deeper level.

The first phase focuses on achieving a working inference pass for the [Mistral 7B model](https://huggingface.co/mistralai/Mistral-7B-v0.1), and the second phase will focus on performance improvements.

# Roadmap

- [x] **Model Loader**
  - [x] **Binary converter** *(scripts/export_mistral.py)*  
    Converts a Hugging Face Mistral model (config, vocab/merges, and weights) into a single standardized binary file that can be fed into the engine. It uses a JSON header to store model metadata, vocabulary, merges, and tensor index information, followed by all model weights packed sequentially as contiguous floating point data.
  - [x] **In-memory loader** *(src/loader/parameters.cpp)*  
    Memory-maps the binary, loads the config and provides direct tensor views.

<br>  

- [x] **Tensor & Ops**
  - [x] Tensor implemented as a view over memory with shape/strides *(src/common/tensor.cpp)*
  - [x] Math operations (e.g. matmul, softmax, RoPE) to be optimized later *(src/backend/cpu/kernels.cpp)*

<br>  

- [x] **Tokenizer**  *(src/tokenizer/tokenizer.cpp)*     
  The tokenizer implements full byte-pair encoding (BPE) compatible with Mistral’s vocabulary. It loads tokenizer.json, builds vocab and merge maps, applies Metaspace pre-tokenization, encodes UTF-8 text by merging token pairs by rank, and supports byte fallback


<br>  

- [x] **Inference State** *(src/common/inference_state.h)*  
  Holds temporary memory and KV cache used during inference

<br>  

- [ ] **Mistral architecture implementation** *(src/model/mistral/modules.cpp)*   
  Each module implementation and test against PyTorch/HF
  - [x] Embedding
  - [x] RMSNorm
  - [x] Rotary Embedding
  - [x] Attention
  - [ ] MLP
  - [ ] LM Head

<br>  

- [ ] **CLI I/O**

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

#### References
- [PyTorch Documentation](https://docs.pytorch.org/docs/stable/index.html)