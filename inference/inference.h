#include "../model/model.h"

struct InferenceState {
    Model& model;

    Tensor x; // hidden state [hidden_size]

    Tensor q;
    Tensor k;
    Tensor v;

    Tensor attn; // [max_seq_len]
    Tensor ctx; // [hidden_size]

    // KV cache
    // Each block stores Tensor of size [n_kv_heads, max_seq_len, head_dim]
    std::vector<std::unique_ptr<Tensor>> k_cache;
    std::vector<std::unique_ptr<Tensor>> v_cache;

    int pos = 0;

    InferenceState(Model& model);
    void block_forward(int b);
    void forward(int token);
    void push_kv(int b);
};