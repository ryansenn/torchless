#include <iostream>
#include "src/loader/parameters.h"
#include "src/model/mistral/modules.h"

int main(){
    std::shared_ptr<Parameters> params = std::make_shared<Parameters>();
    params->load_parameters("../model.bin");
    Arena arena(1024*1024);
    InferenceState infer(params->config);

    RotaryEmbedding::init_freq(infer, params->config);

    infer.pos = 2;
    RotaryEmbedding::forward(infer);

    // q = torch.tensor([[i / 256 for i in range(128)] for j in range(4)])

    Tensor q(arena, {4*128});
    for (int i=0;i<q.shape[0];i++){
        q.data[i] = (i%128) / 256.0f;
    }
    q = q.reshape({1,4,128});

    //q.at({1}).print();

    rope(q, q, infer.cos, infer.sin);

    q.at({0,2}).print();

    return 0;
}
