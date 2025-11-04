#include <iostream>
#include "src/loader/parameters.h"
#include "src/model/mistral/modules.h"

int main(){
    std::shared_ptr<Parameters> params = std::make_shared<Parameters>();
    params->load_parameters("../model.bin");
    InferenceState infer(params->config);

    RotaryEmbedding::init_freq(infer, params->config);

    infer.pos = 3;
    RotaryEmbedding::forward(infer);

    infer.cos.print();
    infer.sin.print();

    return 0;
}
