#include "modules.h"

// NOTE:
// Some modules in this inference engine perform operations in-place on the input tensor
// This means there is no autograd/training support and implementation differs from standard pytorch
// I may choose to refactor this in the future


// https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html
// This one makes a copy, I'm not returning the raw embed weights
// Given a list of indices, we return a 2D tensor containing the tensors at those indices
Tensor Embedding::forward(const std::vector<size_t>& ids){
    size_t num_out_embeddings = ids.size();
    Tensor output({num_out_embeddings, embedding_dim});
    for (size_t i=0; i < num_out_embeddings; i++){
        output.at({i}).copy_from(table.at({ids[i]}));
    }
    return output;
}



// https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.normalization.RMSNorm.html
Tensor RMSNorm::forward(Tensor &x) {
    float squares = 0;

    for(int i =0; i<x.size; i++){
        squares += x.data[i] * x.data[i];
    }

    float rms = sqrt(squares/x.shape[0] + e);

    mul(x, x,1/rms);

    // Element wise mul with g
    for (int i=0; i<x.size; i++){
        x.data[i] = x.data[i] * g.data[i];
    }

    return x;
}

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L290
// This computes the RoPE inverse frequencies
// inv_freq[i] = (1 / rope_theta^(i / (head_dim))) / factor
void RotaryEmbedding::init_freq() {
    for (int i=0;i<inv_freq.size;i+=2){
        float freq = 1.0f / std::pow(rope_theta, float(i)/inv_freq.size);
        inv_freq.data[i] = freq;
        inv_freq.data[i+1] = freq;
    }
}


// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L290
// The forward pass generates cos/sin position encodings for RoPE.
// Take the inv_freq at each position, multiply them by position and apply cos/sin
// Returns 2D tensor containing where
//      out[0] = cos position encodings
//      out[1] = sin position encodings
Tensor RotaryEmbedding::forward(std::vector<size_t> ids) {
    Tensor out({2, ids.size(),head_dim});
    Tensor cos = out.at({0});
    Tensor sin = out.at({1});

    // Looping through each token
    for (size_t i=0;i<cos.shape[0]; i++){
        Tensor tc = cos.at({i});
        Tensor ts = sin.at({i});

        tc.copy_from(inv_freq);
        ts.copy_from(inv_freq);

        float pos = static_cast<float>(ids[i]);

        for (int j=0;j<tc.size;j++){
            tc.data[j] = std::cos(tc.data[j] * pos);
            ts.data[j] = std::sin(ts.data[j] * pos);
        }
    }

    return out;
}
