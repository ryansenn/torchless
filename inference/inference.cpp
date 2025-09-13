#include "inference.h"
#include "math_ops.h"

InferenceState::InferenceState(Model& model) :
    model(model),
    x("x", {model.config.hidden_size}),
    q("q", {model.config.hidden_size}),
    k("k", {model.config.n_kv_heads * model.config.hidden_size / model.config.n_heads}),
    v("v", {model.config.n_kv_heads * model.config.hidden_size / model.config.n_heads}),
    pos(0) {

    // Initialize empty KV cache
    int64_t head_dim = model.config.hidden_size / model.config.n_heads;
    std::vector<int64_t> shape = {model.config.max_seq_len, model.config.n_kv_heads, head_dim};

    for (int i=0; i<model.config.n_layers; i++){
    std::unique_ptr<Tensor> k = std::make_unique<Tensor>("k_cache_" + std::to_string(i), shape);
    std::unique_ptr<Tensor> v = std::make_unique<Tensor>("v_cache_" + std::to_string(i), shape);

    k_cache.push_back(std::move(k));
    v_cache.push_back(std::move(v));
    }
}

// We could probably optimize this by directly matmuling kv directly in cache
void InferenceState::push_kv(int b) {
    // Get K,V
    matmul(k, *model.blocks[b].wk, x);
    matmul(v, *model.blocks[b].wv, x);

    // Append in tensor
    k_cache[b]->at({pos}).copy_from(k);
    v_cache[b]->at({pos}).copy_from(v);
    pos++;
}