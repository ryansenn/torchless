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

    //r.inv_freq.print();

    //sin.at({1}).print();

    cos.at({1}).print();


    return 0;
}
