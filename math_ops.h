//
// Created by Ryan Senoune on 2025-08-23.
//

#ifndef MATH_OPS_H
#define MATH_OPS_H

#include "model/model.h"

// Matrix multiplication
void matmul_impl(float* xout, float* w, float* x, int d, int n);
void matmul(Tensor& xout, Tensor& w, Tensor& x);

// Normalization
void rmsnorm(float* o, float* x, float* g, int n, float eps);
void layernorm(float* o, float* x, float* scale, float* shift, int n, float eps);

// Activation and transformation functions
void softmax(float* o, float* x, int n);

inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
}

inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

inline float clip(float x, float v) {
    return x < -v ? -v : (x > v ? v : x);
}

// Rotary positional encoding
void rope(float* vec, int d, int head_dim, int pos, float theta, int rotary_dim);

#endif // MATH_OPS_H
