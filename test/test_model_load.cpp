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
    // Expected values based on your printout
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

