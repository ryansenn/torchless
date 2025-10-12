//
// Created by Ryan Senoune on 2025-08-23.
//

#ifndef MATH_OPS_H
#define MATH_OPS_H

#include "tensor.h"

// Matrix multiplication
void matmul_impl(float* xout, float* w, float* x, int d, int n);
void matmul(Tensor& xout, Tensor& w, Tensor& x);

void layernorm(float* o, float* x, float* scale, float* shift, int n, float eps);

// Activation and transformation functions
void softmax(float* o, float* x, int n, float t);
inline void softmax(Tensor& o, Tensor& x, int n, float t){
    softmax(o.data, x.data, n, t);
}

inline float gelu(float x);

inline float silu(float x);

inline float clip(float x, float v);

// Rotary positional encoding
void rope(float* vec, int d, int head_dim, int pos, float theta, int rotary_dim);
void rope(Tensor& v, int head_dim, int pos, float theta, int rotary_dim);

#endif // MATH_OPS_H
