//
// Created by Ryan Senoune on 2025-08-23.
//

#ifndef MATH_OPS_H
#define MATH_OPS_H

#include "tensor.h"

float sum(Tensor& x);
void add(Tensor& xout, Tensor& x, float c);
void mul(Tensor& xout, Tensor&x, float c);
void pow(Tensor& xout, Tensor& x, int e);
void sqrt(Tensor& xout, Tensor& x);

void matmul(Tensor& xout, Tensor& w, Tensor& x);
void softmax(Tensor& xout, Tensor x);
void rope(Tensor& xout, Tensor& x);

#endif // MATH_OPS_H
