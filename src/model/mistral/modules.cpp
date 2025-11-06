#include "modules.h"
#include <iostream>


void Embedding::forward(InferenceState& infer, const std::vector<size_t>& ids){
    for (size_t i=0;i<ids.size();i++){
        infer.hidden_state.at({i}).copy_from(table.at({i}));
    }
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

void Attention::forward(InferenceState &infer) {
    // Get q, k, v
    // Rotate q and k


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

void Decoder::forward(InferenceState &infer){
    // Layer norm


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

    // Create layers of decoders
}