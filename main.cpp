#include <iostream>
#include "src/loader/parameters.h"
#include "src/model/mistral/modules.h"

int main(){
    std::shared_ptr<Parameters> params = std::make_shared<Parameters>();
    params->load_parameters("../model.bin");

    RotaryEmbedding r(params->config);
    Tensor emb = r.forward({0,1,2,3});
    Tensor cos = emb.at({0});
    Tensor sin = emb.at({1});

    size_t n_heads = 1;
    size_t seq_len = 4;
    size_t head_dim = cos.shape[1];

    Tensor x({n_heads, seq_len, head_dim});
    Tensor xout({n_heads, seq_len, head_dim});

    for (size_t i = 0; i < x.size; i++)
        x.data[i] = static_cast<float>(i%128) / 256.0f;

    rope(xout, x, cos, sin);

    for (size_t i =0;i<4;i++){
        x.at({0,i}).print();
        xout.at({0,i}).print();
        std::cout << std::endl;
    }

    return 0;
}
