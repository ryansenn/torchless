#pragma once
#include <string>
#include <iostream>

struct Tensor {
    std::string name;
    float* data; // need to free this at some point
    std::vector<int64_t> shape;
    int size;
    std::vector<int64_t> strides;

    uint64_t get_size(){
        uint64_t size = 1;
        for (auto i : shape){
            if (i > 0){
                size *= i;
            }
        }
        return size;
    }

    void init_strides(){
        strides.assign(size, 1);
        int64_t stride = 1;

        for (int i = strides.size() - 2; i >= 0; i--){
            stride *= shape[i+1];
            strides[i] = stride;
        }
    }

    Tensor(std::string name, float* data, std::vector<int64_t> shape): name(std::move(name)), data(data), shape(shape){
        size = get_size();
        init_strides();
    }

    Tensor(std::string name, std::vector<int64_t> shape) : name(std::move(name)), shape(shape){
        size = get_size();
        data = new float[size];
        init_strides();
    }

    // should probably do shape check, type check, etc
    void copy_from(Tensor& tensor){
        memcpy(data, tensor.data, get_size() * sizeof(float));
    }

    void copy_from(float* new_data, int size){
        memcpy(static_cast<void*>(data), new_data, size);
    }

    Tensor at(std::initializer_list<int64_t> idx){
        assert(idx.size() <= shape.size() && "Too many indices for tensor");
        float* new_data;

        int i = 0;
        for (auto v : idx){
            assert(v < shape[i] && "Index out of range");
            new_data += strides[i] * v;
            i++;
        }

        std::vector<int64_t> new_shape(shape.begin() + i, shape.end());

        return Tensor("", new_data, new_shape);
    }

    // Make sure im not copying the tensors
    Tensor(const Tensor&) = delete;

    void check_shape(std::vector<int64_t> expected_shape){
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
};



