#include "modules.h"


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
        size_t j = i % (infer.inv_freq.size / 2);
        infer.cos.data[i] = std::cos(infer.inv_freq.data[j] * infer.pos);
        infer.sin.data[i] = std::sin(infer.inv_freq.data[j] * infer.pos);
    }
}

void Attention::forward(InferenceState &infer) {
    // Get q, k, v
    // Rotate q and k


}

void Decoder::forward(InferenceState &infer){
    // Self attention

    // Residuals
}

// I think I will process one token at a time for now
// And do a rewrite to support multiple tokens
void Model::forward(InferenceState &infer, const std::vector<size_t> &ids) {
    infer.seq_len += ids.size();
    embedding.forward(infer, ids);

    // Should use a causal mask to restrict attention to the window
    // For now, I’ll attend only to tokens in the window
    // On GPU, it’s faster to compute full attention and mask afterward

    // Create layers of decoders
}