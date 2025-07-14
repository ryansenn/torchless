#include <cassert>
#include <string>
#include <iostream>

struct Tensor {
    std::string name;
    float* data;
    std::array<int, 4> shape = {0,0,0,0};

    void check_shape(std::array<int, 4> expected_shape){
        if (this->shape != expected_shape){
            std::cerr << "FATAL: shape mismatch for tensor: " << name << std::endl;
            std::cerr << "Expected: [" << expected_shape[0] << ", "
                      << expected_shape[1] << ", "
                      << expected_shape[2] << ", "
                      << expected_shape[3] << "]" << std::endl;
            std::cerr << "Got: [" << shape[0] << ", "
                      << shape[1] << ", "
                      << shape[2] << ", "
                      << shape[3] << "]" << std::endl;
            assert(false);
        }
    }

    Tensor(std::string name, float* data){
        this->name = name;
        this->data = data;
    }
};

struct Config {
    int vocab_size;
    int hidden_size;
};

struct Model {
    Config config;
    Tensor* token_embedding_table;

    void load(std::string path);
};

struct InferenceState {

};