# Torchless

Torchless is a high-performance C++ LLM inference engine built from scratch. It currently runs the full [Mistral 7B model](https://huggingface.co/mistralai/Mistral-7B-v0.1) on CPU for local text generation. The core development goal is achieving maximum inference speed and supporting advanced architectures like Mistral [Mixture of Experts](https://mistral.ai/news/mixtral-of-experts).

It features custom implementations of tensors, tokenizer and Mistral transformer architecture, all rigorously tested against Hugging Face references. I aim to keep the project's code clear and lightweight as to serve as an educational reference to how a modern LLM runs, layer by layer.

## Running

#### Export with 8-bit quantization

```
python export_mistral.py \
  --model_dir /path/to/Mistral-7B-v0.1 \
  --out ./mistral.bin \
  --quant int8
```

**Note**: The int8-quantized model is the smallest supported variant at ~8 GB. Make sure your system has more than 8 GB of RAM. Systems with 16 GB of RAM run it without problems.

If you run into issues that appear specific to your environment, feel free to open a GitHub issue.

*Will be adding a detailed running guide once quantized inference is stable

## Roadmap

The current work is centered on improving execution speed. I am rewriting some of the transformer modules in C for integrating SIMD and custom CUDA kernels.

<br>

- [x] **Model Loader**
  - [x] **Binary converter** *(scripts/export_mistral.py)*  
    Converts a Hugging Face Mistral model (config, vocab/merges, and weights) into a single standardized binary file, optionally applying quantization. It uses a JSON header to store model metadata, vocabulary, merges, and tensor index information, followed by all model weights packed sequentially as contiguous floating point data.
  - [x] **In-memory loader** *(src/loader/parameters.cpp)*  
    Memory-maps the binary, loads the config and provides direct tensor views.

- [x] **Tensor & Ops**
  - [x] Tensor *(src/common/tensor.cpp)* Implements a strided view over memory supporting f32 and int8, with on-the-fly dequantization during compute
  - [x] Math operations *(src/backend/cpu/kernels.cpp)* e.g. matmul, softmax, RoPE to be optimized later

- [x] **Tokenizer**  *(src/tokenizer/tokenizer.cpp)*     
  The tokenizer implements full byte-pair encoding (BPE) compatible with Mistral’s vocabulary. It loads tokenizer.json, builds vocab and merge maps, applies Metaspace pre-tokenization, encodes UTF-8 text by merging token pairs by rank, and supports byte fallback

- [x] **Inference State** *(src/common/inference_state.h)*  
  Holds temporary memory and KV cache used during inference

- [x] **Mistral architecture implementation** *(src/model/mistral/modules.cpp)*
  - [x] **Embedding** - Looks up initial embedding from token ids
  - [x] **RMSNorm** - computes the RMS over the current hidden_state, normalizes it, and applies the learned gain vector g
  - [x] **Rotary Embedding** - precomputes inverse frequencies from rope_theta and fills cos/sin tensors for RoPE for each position
  - [x] **Attention** - projects hidden_state into Q/K/V, applies RoPE to Q and K, updates the KV cache, runs grouped-query attention over the window, then applies the output projection back into hidden_state
  - [x] **Feedforward MLP** - implements the SwiGLU feedforward: linear projections + SiLU
  - [x] **Layer** - runs norm, attention, and MLP with residuals around each subblock
  - [x] **Model** - embeds input token and runs it through all decoder layers
  - [x] **LM Head** - the final linear layer that projects the last hidden state to the vocabulary size, yielding logits

- [x] **Parity Tests** *(test/mistral)*  
  Comprehensive validation of all inference components (tokenizer, modules, ops) by checking that their outputs match those produced by the Hugging Face Mistral implementation

- [x] **Quantization** *(scripts/quantize.py)*
  - [x] **Per-group symmetric quantization** - splits tensor into groups, for each group, finds max abs value, computes scale and produces quantized weights
  - [ ] **Mixed quantization** - Implement q5_k_m (Use Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K)

- [ ] **CPU Multithreading**

- [ ] **SIMD**

- [x] Token Generation & Sampler
  - [x] Basic text completion with greedy decoding
  - [ ] Multinomial Sampling & temperature scaling

- [ ] **CLI I/O**

- [ ] **Custom CUDA Kernels**

- [ ] **MoE**

# Resources

#### Some of the material that helped me learn the theory or guided me build the engine

##### ML Theory
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [Andrej Karpathy - Let’s build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
- [Rotary Embeddings](https://www.youtube.com/watch?v=V8r__fXx7tU)

##### Systems Internals
- [Edward Z. Yang - PyTorch Internals](https://blog.ezyang.com/2019/05/pytorch-internals/)
- [C++ Vtables](https://shaharmike.com/cpp/vtable-part1/)
- [Andrew Chan - yalm](https://andrewkchan.dev/posts/yalm.html)
- [Arseny Kapoulkine - LLM inference speed of light](https://zeux.io/2024/03/15/llm-inference-sol/)
- [Maxime Labonne - Quantize llama models](https://medium.com/data-science/quantize-llama-models-with-ggml-and-llama-cpp-3612dfbcc172)

##### Reference Implementations
- [Hugging Face - Mistral model](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py)
- [Arseny Kapoulkine - calm](https://github.com/zeux/calm/tree/main)
- [Georgi Gerganov - llama.cpp + ggml](https://github.com/ggml-org/llama.cpp/)
- [Andrej Karpathy - llama2.c quantization](https://github.com/karpathy/llama2.c/blob/master/export.py)