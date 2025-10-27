#include <math.h>
#include "../../common/kernels.h"

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


// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L58
// We perform 2D rotations each pair
// [x', y'] = [x*cosθ - y*sinθ, x*sinθ + y*cosθ]
// The same cos/sin embedding is reused for every head
// Inputs:
//   x: [n_heads, seq_len, head_dim] (q or k)
//   cos, sin: [seq_len, head_dim]
void rope(Tensor& xout, Tensor& x, Tensor& cos, Tensor& sin){
    size_t n_heads = x.shape[0];
    size_t seq_len = x.shape[1];
    size_t head_size = x.shape[2];

    for (int h = 0; h<n_heads; h++){
        for (int p = 0; p < seq_len; p++){
            int start = h*x.strides[0] + p *x.strides[1];
            for (int i = start; i < start+head_size; i+=2){
                float xi = x.data[i];
                float yi = x.data[i+1];
                float c = cos.data[p*cos.strides[0] + (i % head_size)];
                float s = sin.data[p*sin.strides[0] + (i % head_size)];
                xout.data[i] = xi*c - yi*s;
                xout.data[i+1] = xi*s + yi*c;
            }
        }
    }
}


// Not sure if I will be using all of those

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





