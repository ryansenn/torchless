#pragma once
#include <string>
#include <iostream>


enum class DType {
    F32
};

struct Tensor {
    std::string name;
    DType dtype; 
    void* data; // need to free this at some point
    std::array<int64_t, 4> shape = {0,0,0,0};

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

    template <typename T>
    T* get_data(){
        return static_cast<T*>(data);
    }

    // type is hardcoded to float, should be changed if needed
    Tensor(std::string name, void* data): name(std::move(name)), data(data), dtype(DType::F32){}

    // Allocate empty tensor given shape
    Tensor(std::string name, std::array<int64_t, 4> shape) : name(std::move(name)), shape(shape), dtype(DType::F32){
        uint64_t size = 1;
        for (auto i : shape){
            if (i > 0){
                size *= i;
            }
        }

        data = new float[size];
    }
};



