#include "inference.h"
#include "math_ops.h"

InferenceState::InferenceState(Model& model) :
    model(model),
    x("x", {model.config.hidden_size}),
    q("q", {model.config.hidden_size}),
    k("k", {model.config.n_kv_heads * model.config.hidden_size / model.config.n_heads}),
    v("v", {model.config.n_kv_heads * model.config.hidden_size / model.config.n_heads}),
    attn("attn", {model.config.max_seq_len}),
    ctx("context", {model.config.n_heads,model.config.head_dim})
    {

    // Initialize empty KV cache
    int64_t head_dim = model.config.hidden_size / model.config.n_heads;
    std::vector<int64_t> shape = {model.config.n_kv_heads, model.config.max_seq_len, head_dim};

    for (int i=0; i<model.config.n_layers; i++){
    std::unique_ptr<Tensor> k = std::make_unique<Tensor>("k_cache_" + std::to_string(i), shape);
    std::unique_ptr<Tensor> v = std::make_unique<Tensor>("v_cache_" + std::to_string(i), shape);

    k_cache.push_back(std::move(k));
    v_cache.push_back(std::move(v));
    }
}

void InferenceState::block_forward(int b){
    // Get Q for the current token
    matmul(q, *model.blocks[b].wq, x);

    push_kv(b);

    // Reshape by head
    q = q.reshape({model.config.n_heads, model.config.head_dim});

    // Compute attention for each head individually
    for (int h=0; h<model.config.n_heads; h++){
        // Match each q head to a corresponding k head (each k head is re-used 4 times because of grouped-query attention)
        Tensor q_h =  q.at({h}); // [head_dim]

        Tensor k_h = k_cache[b]->at({h / (model.config.n_heads / model.config.n_kv_heads)}); // [max_seq_len x head_dim]
        k_h = k_h.slice1d(pos, model.config.head_dim); // [seq_len x head_dim]

        // [seq_len x head_dim] @ [head_dim] = [seq_len]
        matmul(attn, k_h, q_h); // logits

        // [seq_len]
        softmax(attn, attn, std::min(pos+1, model.config.max_seq_len), sqrt(model.config.head_dim)); // this gives us the weights

        // [seq_len x head_dim]
        Tensor v_h = v_cache[b]->at({h / (model.config.n_heads / model.config.n_kv_heads)});

        // [head_dim]
        Tensor ctx_slice = ctx.at({h});

        // [seq_len] @ [seq_len x head_dim] = [head_dim]
        matmul(ctx_slice, attn, v_h); // Write the attention output directly to corresponding context slice
    }
}

// Trying first minimal implementation of the inference flow
void InferenceState::forward(int token){
    float* embedding = &model.token_embedding_table->data[token * model.config.hidden_size];
    x.copy_from(embedding, model.config.hidden_size);

    rmsnorm(x, x, *model.blocks[0].lm1, model.config.hidden_size);

    // Forward for each block
    for (int i=0; i<model.config.n_layers; i++){
        block_forward(i);
    }

    pos++;
}

// We could probably optimize this by directly matmuling kv directly in cache
// Because we switched layout of KV cache to [n_kv_heads, max_seq_len, head_dim], we are not writing a contiguous chunk anymore
void InferenceState::push_kv(int b) {
    // Get K,V
    matmul(k, *model.blocks[b].wk, x);
    matmul(v, *model.blocks[b].wv, x);

    Tensor kh = k.reshape({model.config.n_kv_heads, model.config.head_dim});
    Tensor vh = v.reshape({model.config.n_kv_heads, model.config.head_dim});

    for (int h=0;h<model.config.n_kv_heads; h++){
        // Append in tensor
        k_cache[b]->at({h, pos}).copy_from(kh.at({h}));
        v_cache[b]->at({h, pos}).copy_from(vh.at({h}));
    }
}