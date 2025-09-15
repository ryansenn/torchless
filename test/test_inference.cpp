#include "context.h"

int test_kv_cache() {
    InferenceState inferenceState(get_model());
    int hidden_size = inferenceState.model.config.hidden_size;

    // hijack the hidden state with a bunch of ones
    float ones[hidden_size];
    for (int i = 0; i < hidden_size; i++) ones[i] = 1.0f;
    inferenceState.x.copy_from(ones, hidden_size * sizeof(float));

    // push all layers
    for (int i=0;i<inferenceState.model.config.n_layers; i++){
        inferenceState.push_kv(i);
    }
    inferenceState.pos++;

    if (inferenceState.pos != 1) {
        std::cout << "pos mismatch after push: got " << inferenceState.pos
                  << ", want 1" << std::endl;
        return 1;
    }

    // check layer 0
    inferenceState.push_kv(0);
    if (!equals(inferenceState.k_cache[0]->data[0], 0.00145736f)) {
        std::cout << "k_cache[0]->data[0] mismatch: got "
                  << inferenceState.k_cache[0]->data[0] << ", want 0.00145736" << std::endl;
        return 1;
    }
    if (!equals(inferenceState.k_cache[0]->data[1023], -0.04015780f)) {
        std::cout << "k_cache[0]->data[1023] mismatch: got "
                  << inferenceState.k_cache[0]->data[1023] << ", want -0.04015780" << std::endl;
        return 1;
    }

    // check at layer 31
    inferenceState.push_kv(31);
    if (!equals(inferenceState.k_cache[31]->data[0], 0.11939027f)) {
        std::cout << "k_cache[31]->data[0] mismatch: got "
                  << inferenceState.k_cache[31]->data[0] << ", want 0.11939027" << std::endl;
        return 1;
    }
    if (!equals(inferenceState.k_cache[31]->data[1023], -0.11372215f)) {
        std::cout << "k_cache[31]->data[1023] mismatch: got "
                  << inferenceState.k_cache[31]->data[1023] << ", want -0.11372215" << std::endl;
        return 1;
    }

    // push again, should populate pos 1
    for (int i=0;i<inferenceState.model.config.n_layers; i++){
        inferenceState.push_kv(i);
    }
    inferenceState.pos++;

    if (!equals(inferenceState.k_cache[0]->at({1}).data[0], 0.00145736f)) {
        std::cout << "k_cache[0]->data[" << 1024 + 0
                  << "] mismatch at pos1: got " << inferenceState.k_cache[0]->at({1}).data[0]
                  << ", want 0.00145736" << std::endl;
        return 1;
    }
    if (!equals(inferenceState.k_cache[0]->at({1}).data[1023], -0.04015780f)) {
        std::cout << "k_cache[0]->data[" << 1024 + 1023
                  << "] mismatch at pos1: got " << inferenceState.k_cache[0]->at({1}).data[1023]
                  << ", want -0.04015780" << std::endl;
        return 1;
    }

    return 0;
}

RegisterTest reg_test_kv_cache("test kv cache", &test_kv_cache);