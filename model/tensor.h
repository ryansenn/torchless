#pragma once
#include <string>
#include <iostream>

struct Tensor {
    std::string name;
    float* data; // need to free this at some point
    std::array<int64_t, 4> shape = {0,0,0,0};

    uint64_t get_size(){
        uint64_t size = 1;
        for (auto i : shape){
            if (i > 0){
                size *= i;
            }
        }
        return size;
    }

    Tensor(std::string name, float* data): name(std::move(name)), data(data){}

    Tensor(std::string name, std::array<int64_t, 4> shape) : name(std::move(name)), shape(shape){
        data = new float[get_size()];
    }

    // should probably do shape check, type check, etc
    void copy_from(Tensor& tensor){
        memcpy(data, tensor.data, get_size() * sizeof(float));
    }

    void copy_from(float* new_data, int size){
        memcpy(static_cast<void*>(data), new_data, size);
    }

    void check_shape(std::array<int64_t, 4> expected_shape){
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

    // Make sure im not copying the tensors
    Tensor(const Tensor&) = delete;
};



