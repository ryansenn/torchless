#include "../../common/kernels.h"
#include "../../loader/parameters.h"
#include "inference_state.h"


// https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html
struct Embedding {
    Tensor<float> table;
    size_t num_embeddings;
    size_t embedding_dim;
    Embedding(Tensor<float>& table) : table(table), num_embeddings(table.shape[0]), embedding_dim(table.shape[1]) {}
    void forward(InferenceState& infer, size_t token_id);
};

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L268
struct RotaryEmbedding {
    static void init_freq(InferenceState& infer, Config& config);
    static void forward(InferenceState& infer);
};

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L58
struct RMSNorm {
    Tensor<float> g;
    float e = 1e-6f;

    RMSNorm(const Tensor<float>& g) : g(g) {}
    void forward(InferenceState& infer);
};

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L123
struct Attention {
    Tensor<float> q_proj;
    Tensor<float> k_proj;
    Tensor<float> v_proj;
    Tensor<float> o_proj;

    Attention(const Tensor<float>& q_proj,
              const Tensor<float>& k_proj,
              const Tensor<float>& v_proj,
              const Tensor<float>& o_proj)
            : q_proj(q_proj),
              k_proj(k_proj),
              v_proj(v_proj),
              o_proj(o_proj) {}

    void forward(InferenceState& infer);
};


// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L35
struct MLP {
    Tensor<float> down_proj;
    Tensor<float> gate_proj;
    Tensor<float> up_proj;

    MLP(const Tensor<float>& down_proj, const Tensor<float>& gate_proj, const Tensor<float>& up_proj) : down_proj(down_proj), gate_proj(gate_proj), up_proj(up_proj){}

    void forward(InferenceState& infer);
};


// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L206
struct Layer {
    RMSNorm input_norm;
    RMSNorm output_norm;
    Attention attn;
    MLP mlp;

    Layer(const std::unordered_map<std::string, Tensor<float>>& w) :

                                input_norm(w.at("input_layernorm.weight")),
                                output_norm(w.at("post_attention_layernorm.weight")),

                                attn(w.at("self_attn.q_proj.weight"),
                                     w.at("self_attn.k_proj.weight"),
                                     w.at("self_attn.v_proj.weight"),
                                     w.at("self_attn.o_proj.weight")),

                                mlp(w.at("mlp.down_proj.weight"),
                                     w.at("mlp.gate_proj.weight"),
                                     w.at("mlp.up_proj.weight"))
                                {}


    void forward(InferenceState& infer);
};


// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L414
struct LMHead {
    Tensor<float> lm_head; // [4096, vocab_size]

    LMHead(std::shared_ptr<Parameters> params) : lm_head(params->global_weights.at("lm_head.weight")) {}

    void forward(InferenceState& infer);
};


// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L334
struct Model {
    Embedding embedding;
    RMSNorm norm;
    LMHead lmHead;
    std::vector<Layer> layers;

    Model(std::shared_ptr<Parameters> params) : embedding(params->global_weights.at("model.embed_tokens.weight")), norm(params->global_weights.at("model.norm.weight")), lmHead(params){
        for (int i=0;i<params->config.n_layers; i++){
            layers.emplace_back(params->layer_weights[i]);
        }
    }

    void forward(InferenceState& infer, size_t token_id);
};




