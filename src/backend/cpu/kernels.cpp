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

// Here we compute the max and subtract it from each logit to avoid overflow,
// keeping the relative ratios unchanged (softmax is shift-invariant)
void softmax(Tensor& xout, Tensor x){
    float maxv = x.max();
    float total = 0;
    for (int i=0; i<x.size; i++){
        xout.data[i] = std::expf(x.data[i] - maxv);
        total += xout.data[i];
    }
    mul(xout, xout, 1/total);
}

// RoPE rotates tokens based on their position in a sequence.
// The idea is that tokens that are close to each other
// should have a smaller angle difference between them

// We perform 2D rotations each pair in tensor x
// The higher the position in sequence, the more we rotate
// The rotation also shrinks exponentially as we advance in the tensor
void rope(Tensor& xout, Tensor& x){

}





