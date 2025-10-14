#include <math.h>
#include "../../common/kernels.h"

float sum(Tensor& x){
    float r = 0.0f;
    for (int i=0; i<x.size; i++){
        r += x.data[i];
    }
    return r;
}

void add(Tensor& xout, Tensor& x, float c){
    for (int i = 0; i < x.size; i++) {
        xout.data[i] = x.data[i] + c;
    }
}

void mul(Tensor& xout, Tensor& x, float c) {
    for (int i = 0; i < x.size; i++) {
        xout.data[i] = x.data[i] * c;
    }
}

void pow(Tensor& xout, Tensor& x, int e){
    for (int i=0; i<x.size; i++){
        xout.data[i] = pow(x.data[i], e);
    }
}

void sqrt(Tensor& xout, Tensor& x){
    for (int i=0; i<x.size; i++){
        xout.data[i] = sqrt(x.data[i]);
    }
}

// This only works for
// W (d,n) @ x (n,) = xout (d,)
void matmul(Tensor& xout, Tensor& w, Tensor& x){
    size_t d = w.shape[0];
    size_t n = w.shape[1];

    for (int i=0; i<d; i++){
        xout.data[i] = 0;
        for (int j=0; j<n; j++){
            xout.data[i] += w.data[i*n+j] * x.data[j];
        }
    }
}




