#include <iostream>
#include "src/loader/parameters.h"
#include "src/model/mistral/modules.h"

int main(){
    std::shared_ptr<Parameters> params = std::make_shared<Parameters>();
    params->load_parameters("../model.bin");
    Arena arena(1024*1024);
    InferenceState infer(params->config);

    auto& w = params->layer_weights[0];
    Attention attn(w.at("self_attn.q_proj.weight"), w.at("self_attn.k_proj.weight"), w.at("self_attn.v_proj.weight"));

    Embedding embedding(params->global_weights.at("model.embed_tokens.weight"));
    embedding.forward(infer, {50});
    attn.forward(infer);

    infer.q.print();

    return 0;
}
