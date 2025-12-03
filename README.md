# Torchless

Torchless is a custom-built LLM inference engine written entirely from scratch. It currently runs [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) on CPU for local text completion.

The current goal of this project is to reach maximum inference speed and support new state-of-the-art [Mistral 3](https://mistral.ai/news/mistral-3) architectures.

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
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

#### Run
```
./torchless ../mistral.bin "Paris is the capital of"
```

If you run into issues that appear specific to your environment, feel free to open a GitHub issue.

# Project Roadmap

The initial phase of the project focused on correctness and building the necessary backend infrastructure for completing an inference pass. This involved implementing a model loader, tensor/math utilities, tokenizer, transformer architecture and verifying everything against Hugging Face.

Current development focuses on performance optimization and model expansion:
- Ongoing rewrite of slow code sections
- Implementing CPU SIMD instructions and custom CUDA kernels (primary focus)
- Supporting [Ministral 3 3B](https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512)

More detailed roadmap can be found [here](roadmap.md).

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