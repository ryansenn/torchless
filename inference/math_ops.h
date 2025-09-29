//
// Created by Ryan Senoune on 2025-08-23.
//

#ifndef MATH_OPS_H
#define MATH_OPS_H

#include "../model/model.h"

// Matrix multiplication
void matmul_impl(float* xout, float* w, float* x, int d, int n);
void matmul(Tensor& xout, Tensor& w, Tensor& x);

void rowvec_matmul_impl(float* xout, float* x, float* w, int d, int n);
void rowvec_matmul(Tensor& xout, Tensor& x, Tensor& w);

// Normalization
void rmsnorm(float* o, float* x, float* g, int n, float eps);
inline void rmsnorm(Tensor& o, Tensor& x, Tensor& g, int n){
    rmsnorm(o.data, x.data, g.data, n , 0); // supposed to use eps > 0 ? change if failure
}

void layernorm(float* o, float* x, float* scale, float* shift, int n, float eps);

// Activation and transformation functions
void softmax(float* o, float* x, int n, float t);
inline void softmax(Tensor& o, Tensor& x, int n, float t){
    softmax(o.data, x.data, n, t);
}

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
void rope(Tensor& v, int head_dim, int pos, float theta, int rotary_dim);

#endif // MATH_OPS_H
