#include <math.h>
#include <iostream>

// W (d,n) @ x (n,) = xout (d,)
void matmul(float* xout, float* x, float* w, int n, int d){
    for (int i=0; i<d; i++){
        float res = 0;

        for (int j=0; j<n; j++){
            res += w[i*n + j] * x[j];
        }
        xout[i] = res;
    }
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
void softmax(float* o, float* x, int n){
    float total = 0;
    for (int i=0;i<n;i++){
        o[i] = std::expf(x[i]);
        total += o[i];
    }

    for (int i=0;i<n;i++){
        o[i] /= total;
    }
}

// softmax, gelu, silu, clip, rope, attn, block, forward