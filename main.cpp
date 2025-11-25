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

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>\n";
        return 1;
    }

    Arena arena(1024*1024);
    Tensor w(arena, {1, 2, 3, 4, 5, 6}, {2, 3});


    /*
    std::string model_path = argv[1];

    std::shared_ptr<Parameters> params = std::make_shared<Parameters>();
    params->load_parameters(model_path);

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
     */

    return 0;
}
