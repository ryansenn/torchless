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

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L334
struct Model {
    std::shared_ptr<Parameters> params;

    Model(std::shared_ptr<Parameters> params) : params(params) {}

    void forward(InferenceState& infer, const std::vector<size_t>& ids);
};

