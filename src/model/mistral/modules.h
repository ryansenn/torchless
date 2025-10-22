#include "../../common/kernels.h"
#include "../../loader/parameters.h"

// NOTE:
// All modules in this inference engine perform operations in-place on the input tensor
// This means there is no autograd/training support and implementation differs from standard pytorch
// I may choose to refactor this in the future


// https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html
struct Embedding {
    Tensor table;
    size_t num_embeddings;
    size_t embedding_dim;
    Embedding(Tensor& table) : table(table), num_embeddings(table.shape[0]), embedding_dim(table.shape[1]) {}
    Tensor forward(const std::vector<size_t>& ids);
};


// https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.normalization.RMSNorm.html
struct RMSNorm {
    Tensor g;
    float e = 1e-6f;
    RMSNorm(Tensor& g) : g(g) {}
    Tensor forward(Tensor& x);
};

// Build cos/sin embeddings for RoPE
// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L268
struct RotaryEmbedding {
    Tensor cos;
    Tensor sin;

    float rope_theta;
    size_t head_dim;
    Tensor inv_freq;

    // https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py#L178
    void init_freq();

    RotaryEmbedding(Config& config) : rope_theta(config.rope_theta), head_dim(config.head_dim), inv_freq({head_dim}){
        init_freq();
    }

    Tensor forward(std::vector<size_t> ids);
};
