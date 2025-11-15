#include "modules.h"
#include <iostream>

// Assumes only 1 id
void Embedding::forward(InferenceState& infer, const std::vector<size_t>& ids){
    infer.hidden_state.copy_from(table.at({ids[0]}));
}

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L290
// Computes RoPE inverse frequencies
// inv_freq[i] = (1 / rope_theta^(i / (head_dim))) / factor
void RotaryEmbedding::init_freq(InferenceState& infer, Config& config) {
    for (int i=0;i<infer.inv_freq.size;i++){
        float freq = 1.0f / std::pow(config.rope_theta, float(i)/infer.inv_freq.size);
        infer.inv_freq.data[i] = freq;
    }
}

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L319
// The forward pass generates cos/sin position encodings for RoPE.
// Take the inv_freq at each position, multiply them by position and apply cos/sin
void RotaryEmbedding::forward(InferenceState& infer){
    for (size_t i=0; i<infer.cos.size; i++){
        infer.cos.data[i] = std::cos(infer.inv_freq.data[i % infer.inv_freq.size] * infer.pos);
        infer.sin.data[i] = std::sin(infer.inv_freq.data[i % infer.inv_freq.size] * infer.pos);
    }
}

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L195
// x*g / sqrt(sum(x^2) + e)
void RMSNorm::forward(InferenceState& infer) {
    float squares = 0;

    for(int i =0; i<infer.hidden_state.size; i++){
        squares += infer.hidden_state.data[i] * infer.hidden_state.data[i];
    }

    float rms = sqrt(squares/infer.hidden_state.shape[0] + e);

    mul(infer.hidden_state, infer.hidden_state,1/rms);

    // Element wise mul with g
    for (int i=0; i<infer.hidden_state.size; i++){
        infer.hidden_state.data[i] = infer.hidden_state.data[i] * g.data[i];
    }
}

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L140
void Attention::forward(InferenceState &infer) {
    // Get q, k, v
    // [proj, hidden_size] @ [hidden_size, 1] = [proj]
    matmul(infer.q_state, q_proj, infer.hidden_state);
    matmul(infer.k_state, k_proj, infer.hidden_state);
    matmul(infer.v_state, v_proj, infer.hidden_state);

    // Populate cos/sin embeddings
    RotaryEmbedding::forward(infer);

    // Rotate Q,K
    rope(infer.q_state, infer.q_state, infer.cos, infer.sin);
    rope(infer.k_state, infer.k_state, infer.cos, infer.sin);

    // Push KV to cache
    infer.push_kv();

    // Perform attention with tokens in window
    // softmax ( QK^t / sqrt(head_dim) ) * V
    // TODO: Try repeating K and do 1 matmul instead of 32 individual ones

    // Reuse each KV head 4 times
    for (size_t h=0; h<infer.config.n_heads; h++){
        // [seq_len, 128] @ [128]
        Tensor q_head = infer.q_state.at({h}); // [128]
        Tensor k_head = infer.k_cache.at({h/4}).reshape({infer.pos+1, infer.config.head_dim}); // [seq_len, 128]
        Tensor score_head = infer.scores.at({h}).reshape({infer.pos+1});

        // KQ
        matmul(score_head, k_head, q_head);
        // Divide by dk
        mul(score_head, score_head, 1/sqrt(infer.config.head_dim));
        // Softmax
        softmax(score_head, score_head); // [seq_len]

        Tensor v_head = infer.v_cache.at({h/4}).reshape({infer.pos+1, infer.config.head_dim});  // [seq_len, 128]
        Tensor context_head = infer.context.at({h});

        // score_head [seq_len] @ v_head [seq_len, head_dim]
        row_matmul(context_head, score_head, v_head);
    }

}

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L215
void Layer::forward(InferenceState &infer){
    // Layer norm
    norm.forward(infer);

    // Self attention

    // Residuals
}

// I think I will process one token at a time for now
// And do a refactor/rewrite to support multiple tokens
void Model::forward(InferenceState &infer, const std::vector<size_t> &ids) {
    infer.seq_len += ids.size();
    embedding.forward(infer, ids);

    // Should use a causal mask to restrict attention to the window
    // For now, I’ll attend only to tokens in the window
    // will compute full attention and mask afterward

    // forward each layer
}