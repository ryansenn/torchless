//
// Created by Ryan Senoune on 2025-08-23.
//

#ifndef MATH_OPS_H
#define MATH_OPS_H

#include "tensor.h"


float sum(Tensor& x);
void mul(Tensor& xout, Tensor&x, float c);
void pow(Tensor& xout, Tensor& x, int e);
void sqrt(Tensor& xout, Tensor& x);

// Matrix multiplication
void matmul(float* xout, float* w, float* x, int d, int n);
inline void matmul(Tensor& xout, Tensor& w, Tensor& x){
    matmul(xout.data, w.data, x.data, w.shape[0], w.shape[1]);
}

void layernorm(float* o, float* x, float* scale, float* shift, int n, float eps);

// Activation and transformation functions
void softmax(float* o, float* x, int n, float t);
inline void softmax(Tensor& o, Tensor& x, int n, float t){
    softmax(o.data, x.data, n, t);
}

#endif // MATH_OPS_H
