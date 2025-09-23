#include <math.h>
#include "../model/model.h"
#include <iostream>

// W (d,n) @ x (n,) = xout (d,)
void matmul_impl(float* xout, float* w, float* x, int d, int n){
    for (int i=0; i<d; i++){
        float res = 0;

        for (int j=0; j<n; j++){
            res += w[i*n + j] * x[j];
        }
        xout[i] = res;
    }
}

void matmul(Tensor& xout, Tensor& w, Tensor& x){
    matmul_impl(xout.data, w.data, x.data, w.shape[0], w.shape[1]);
}

// out(d) = x(n) @ W(n,d)
// row-vector times matrix, same as (W^T @ x_col)
void rowvec_matmul_impl(float* xout, float* x, float* w, int n, int d){
    // column by column
    for(int i=0;i<d;i++){
        float res = 0;

        // row by row
        for(int j=0;j<n;j++){
            res += x[j] * w[j*d + i];
        }
        xout[i] = res;
    }
}

void rowvec_matmul(Tensor& xout, Tensor& x, Tensor& w){
    matmul_impl(xout.data, w.data, x.data, w.shape[0], w.shape[1]);
}

void rmsnorm(float* o, // output
             float* x, // input
             float* g, // scale per element
             int    n, // size
             float eps) // epsilon 
{
    float mean_squared = 0;
    for (int i=0;i<n;i++){
        mean_squared += x[i]*x[i];
    }
    mean_squared = mean_squared/n;
    float m = 1/sqrtf(mean_squared + eps);

    for (int i=0; i<n; i++){
        o[i] = x[i] * g[i] * m;
    }

}

void layernorm(float* o, float* x, float* scale, float* shift, int n, float eps){

    float u = 0;
    for (int i=0;i<n;i++){
        u += x[i];
    }
    u /= n;

    float s = 0;
    float d;
    for (int i=0;i<n;i++){
        d = x[i]-u;
        s += d*d;
    }
    s /= n;
    s = sqrtf(s);
    float s2 = 1/(s+eps);

    for (int i=0;i<n;i++){
        o[i] = (x[i] - u) * s2 * scale[i] + shift[i];
    }
}

// If there is numerical instability, try to substract max value from each xi before exp
void softmax(float* o, float* x, int n, float t){
    float total = 0;
    for (int i=0;i<n;i++){
        o[i] = std::expf(x[i]/t);
        total += o[i];
    }

    for (int i=0;i<n;i++){
        o[i] /= total;
    }
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

void rope(float* vec, int d, int head_dim, int pos, float theta, int rotary_dim) {
    for (int i = 0; i < d; i += 2) {
        int j_head = i % head_dim;
        float freq = 0.f;
        if (j_head < rotary_dim) {
            freq = 1.0f / powf(theta, static_cast<float>(j_head) / static_cast<float>(rotary_dim));
        }
        float val = pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);

        float v0 = vec[i];
        float v1 = vec[i + 1];
        vec[i] = v0 * fcr - v1 * fci;
        vec[i + 1] = v0 * fci + v1 * fcr;
    }
}

