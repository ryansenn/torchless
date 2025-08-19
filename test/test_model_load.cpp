#include "../model/model.h"
#include "registry.h"

Model& get_model() {
    static Model model;
    static bool loaded = false;
    if (!loaded) {
        model.load("../model.bin");
        loaded = true;
    }
    return model;
}

int test_model_load_metadata(){
    Model m = get_model();

    if (!m.config) {
        std::cout << "config is null" << std::endl;
        return 1;
    }
    if (m.config->hidden_size != 4096) {
        std::cout << "hidden_size mismatch: got " << m.config->hidden_size
                  << ", want 4096" << std::endl;
        return 1;
    }
    if (m.config->vocab_size != 32000) {
        std::cout << "vocab_size mismatch: got " << m.config->vocab_size
                  << ", want 32000" << std::endl;
        return 1;
    }

    return 0;
}

static RegisterTest load_metadata("load metadata",&test_model_load_metadata);


int test_model_load_embedding() {
    Model& m = get_model();

    if (!m.token_embedding_table) {
        std::cout << "token_embedding_table is null" << std::endl;
        return 1;
    }

    float* data = static_cast<float*>(m.token_embedding_table->data);
    float got = data[0];
    float expected = -2.18642e-36f;

    if (std::fabs(got - expected) > 1e-40f) {
        std::cout << "embedding[0] mismatch: got " << got
                  << ", want " << expected << std::endl;
        return 1;
    }

    return 0;
}

static RegisterTest load_embedding("load embedding", &test_model_load_embedding);