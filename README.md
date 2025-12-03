# Torchless

Torchless is an LLM inference engine built entirely from scratch. It currently runs the [Mistral 7B model](https://huggingface.co/mistralai/Mistral-7B-v0.1) on CPU for local text completion.



The goal of this project is to reach maximum inference speed and support the new state-of-the-art [Mistral 3](https://mistral.ai/news/mistral-3) architectures.

# Running

#### Download Mistral 7B v0.1, Torchless and nhlohmann JSON library

```
git clone https://huggingface.co/mistralai/Mistral-7B-v0.1
git clone https://github.com/ryanssenn/torchless.git
cd torchless
curl -L https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp -o src/common/json.hpp
```

#### (Optional) Create Python virtual environment and download libraries

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Export the model with 8-bit quantization

```
python3 export_mistral.py \
  --model_dir ../Mistral-7B-v0.1 \
  --out ./mistral.bin \
  --quant f32
```

#### Compile project

```
mkdir build
cd build
cmake ..
cmake --build .
```

#### Run
```
torchless mistral.bin "Paris is the capital of"
```


If you run into issues that appear specific to your environment, feel free to open a GitHub issue.

# Project Roadmap

The initial phase of the project focused on correctness and building the necessary backend infrastructure for completing an inference pass. This involved implementing a model loader, tensor/math utilities, tokenizer, transformer architecture and verifying everything against Hugging Face.

Work is now focused on:
- Rewriting slow sections of the code
- Adding SIMD paths on CPU and custom CUDA kernels
- Supporting Mistral 3 newest architectures starting with [Ministral 3](https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512)
- Building a small terminal chat interface

## Loading...

### Model Loader
- [x] **Model binary converter** *(export_mistral.py)*  
  Converts a Hugging Face Mistral model (config, vocab/merges, and weights) into a single standardized binary file, optionally applying quantization. It uses a JSON header to store model metadata, vocabulary, merges, and tensor index information, followed by all model weights packed sequentially as contiguous floating point data.
- [x] **In-memory loader** *(src/loader/parameters.cpp)*  
  Memory-maps the binary, loads the config and provides direct tensor views.

### Tensor & Ops
- [x] **Tensor** *(src/common/tensor.cpp)*  
  Implements a strided view over memory supporting f32 and int8, with on-the-fly dequantization during compute
- [x] **Math operations** *(src/backend/cpu/kernels.cpp)*  
  Implementation of matmul, softmax and RoPE to be optimized later

## Text In, Tokens Out

### Tokenizer
- [x] **Tokenizer** *(src/tokenizer/tokenizer.cpp)*  
  Implements full byte-pair encoding (BPE) compatible with Mistral’s vocabulary. It loads tokenizer.json, builds vocab and merge maps, applies Metaspace pre-tokenization, encodes UTF-8 text by merging token pairs by rank, and supports byte fallback

### Token Generation & Sampler
- [x] Basic text completion with greedy decoding
- [ ] Multinomial Sampling & temperature scaling

### CLI I/O
- [ ] Build a terminal chat inferface

## Core Transformer
The architecture *(src/model/mistral/modules.cpp)* is broken into independent C++ structs using a shared inference state to manage memory and cache. The implementation encodes relative positions using rotary embeddings (RoPE), applies gated SwiGLU in the feed-forward layers, and utilizes grouped-query attention (GQA) which assigns multiple query heads to share a single key-value head pair.

### Inference State
- [x] Temporary memory and cache *(src/common/inference_state.h)* used to hold all intermediate tensors for a single token's computation during the forward pass

### Modules
- [x] **Embedding** - Looks up initial embedding from token and copies it to `infer.hidden_state`
- [x] **RMSNorm** - Initializes inverse frequencies based on rope theta and generates cosine/sine tables dynamically based on the current `infer.pos`
- [x] **Rotary Embedding** - precomputes inverse frequencies from rope_theta and fills cos/sin tensors for RoPE for each position
- [x] **Attention** - Projects to Q/K/V, applies rotary embeddings to Q/K, pushes to the KV cache, runs the grouped-query attention mechanism (reusing KV heads 4x), and projects the result.
- [x] **Feedforward MLP** - Implements the SwiGLU feedforward: linear projections + SiLU
- [x] **Layer** - Runs norm, attention, and MLP with residuals around each subblock
- [x] **Model** - Embeds input token and runs it through all decoder layers
- [x] **LM Head** - Projects the final `infer.hidden_state` onto the vocabulary dimension to populate `infer.logits`

### Parity Tests
- [x] Comprehensive validation in *(test/mistral)* of all inference components (tokenizer, modules, ops) by checking that their outputs match those produced by the Hugging Face Mistral implementation

## Gotta go fast

### Quantization
- [ ] Support fp8 with a cast `tensor.to(torch.float8_e5m2)` during model export
- [x] **Per-group symmetric quantization** - splits tensor into groups, for each group, finds max abs value, computes scale and produces quantized weights

### CPU Multithreading
- [ ] Todo

### SIMD
- [ ] Todo

### Custom CUDA Kernels
- [ ] Todo


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