#include "../../common/kernels.h"
#include "../../loader/parameters.h"
#include "inference_state.h"


// https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html
struct Embedding {
    Tensor table;
    size_t num_embeddings;
    size_t embedding_dim;
    Embedding(Tensor& table) : table(table), num_embeddings(table.shape[0]), embedding_dim(table.shape[1]) {}
    void forward(InferenceState& infer, const std::vector<size_t>& ids);
};

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L268
struct RotaryEmbedding {
    static void init_freq(InferenceState& infer, Config& config);
    static void forward(InferenceState& infer);
};

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L123
struct Attention {
    RotaryEmbedding rotary_emb;

    void forward(InferenceState& infer);
};

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L58
struct RMSNorm {
    Tensor g;
    float e = 1e-6f;

    RMSNorm(Tensor& g) : g(g) {}
    void forward(InferenceState& infer);
};

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L206
struct Decoder {
    RMSNorm norm;

    Decoder(Tensor& g) : norm(g){}
    void forward(InferenceState& infer);
};


// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L334
struct Model {
    std::shared_ptr<Parameters> params;
    Embedding embedding;

    Model(std::shared_ptr<Parameters> params) : params(params),
                                                embedding(*params->global_weights["model.embed_tokens.weight"])
                                                {}

    void forward(InferenceState& infer, const std::vector<size_t>& ids);
};

