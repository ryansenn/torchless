#include "../../common/kernels.h"

// NOTE:
// All modules in this inference engine perform operations in-place on the input tensor
// This design eliminates memory copies and should make inference more efficient
// This means there is no autograd/training support and implementation differs from standard pytorch
// I may choose to refactor this in the future

// https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.normalization.RMSNorm.html
struct RMSNorm {
    Tensor g;
    float e = 1e-6f;
    RMSNorm(Tensor& g) : g(g) {}
    Tensor forward(Tensor& x);
};