#include <math.h>
#include "../../common/kernels.h"

float sum(Tensor& x){
    float r = 0.0f;
    for (int i=0; i<x.size; i++){
        r += x.data[i];
    }
    return r;
}

void mul(Tensor& xout, Tensor&x, float c) {
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

// W (d,n) @ x (n,) = xout (d,)
void matmul(float* xout, float* w, float* x, int d, int n){
    for (int i=0; i<d; i++){
        float res = 0;

        for (int j=0; j<n; j++){
            res += w[i*n + j] * x[j];
        }
        xout[i] = res;
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



