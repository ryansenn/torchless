# torchless

A C++ inference engine that runs LLMs directly from model weights.

I have decided to completely rewrite the Model/InferenceState into a unified Model which is closer to a Pytorch-like implementation.

- Each model will keep track of its own state, and reference external weights that will be loaded once, so KV cache will be moved to Model
- A model will hold an abstract list of Block, each Block will hold an implementation of the Attention, you can call forward() on a block or any submodule
- Each module will be unit tested against Pytorch to ensure I'm going in the right direction

Resources: 

- [Attention is all you need](https://arxiv.org/pdf/1706.03762)
- [Hugging face mistral implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py)
