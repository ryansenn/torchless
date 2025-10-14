#include "modules.h"

// NOTE:
// All modules in this inference engine perform operations in-place on the input tensor
// This design eliminates memory copies and should make inference more efficient
// This means there is no autograd/training support and implementation differs from standard pytorch
// I may choose to refactor this in the future


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