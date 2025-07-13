#include <cassert>

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

struct Model {
    float* token_embedding_table;
};

struct InferenceState {

};