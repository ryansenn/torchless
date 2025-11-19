#include <iostream>
#include "src/loader/parameters.h"
#include "src/model/mistral/modules.h"

size_t sample_max(InferenceState& infer){
    float max_val = infer.logits.data[0];
    size_t res = 0;
    for (size_t i = 0;i < infer.config.vocab_size; i++){
        if (infer.logits.data[i] > max_val){
            res = i;
            max_val = infer.logits.data[i];
        }
    }

    return res;
}

size_t generate(Model& model, InferenceState& infer, size_t token){
    model.forward(infer, token);
    return sample_max(infer);
}

int main(){
    std::shared_ptr<Parameters> params = std::make_shared<Parameters>();
    params->load_parameters("../model.bin");

    InferenceState infer(params->config);
    Model model(params);

    const std::string text = "Hello are you alive?";
    std::vector<uint32_t> got = params->tokenizer.encode(text);

    for (int i=0;i<got.size()-1;i++){
        generate(model, infer, got[i]);
    }

    size_t t = got[got.size()-1];
    for (int i = 0; i<50;i++){
        t = generate(model, infer, t);
        std::cout << t;
    }

    std::cout << std::endl;

    return 0;
}
