#include "../model/model.h"

struct InferenceState {
    Model& model;

    Tensor x; // [hidden_size]

    Tensor q;
    Tensor k;
    Tensor v;

    // KV cache
    // Each block stores Tensor of size [max_seq_len, n_kv_heads, head_dim]
    std::vector<std::unique_ptr<Tensor>> k_cache;
    std::vector<std::unique_ptr<Tensor>> v_cache;

    int pos;

    InferenceState(Model& model);
    void push_kv(int b);
};