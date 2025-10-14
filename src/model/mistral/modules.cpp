#include "modules.h"

// NOTE:
// Some modules in this inference engine perform operations in-place on the input tensor
// This means there is no autograd/training support and implementation differs from standard pytorch
// This design eliminates memory copies and should make inference more efficient
// I may choose to refactor this in the future


// https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html
// This one makes a copy, I'm not returning the raw embed weights
// Given a list of indices, we return a 2D tensor containing the tensors at those indices
Tensor Embedding::forward(const std::vector<size_t>& idx){
    size_t num_out_embeddings = idx.size();
    Tensor output({num_out_embeddings, embedding_dim});
    for (size_t i=0; i < num_out_embeddings; i++){
        output.at({i}).copy_from(table.at({idx[i]}));
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