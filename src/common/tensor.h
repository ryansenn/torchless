#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <initializer_list>

// TODO: Tensors will be views, need to remove all memory allocations

struct Tensor {
    float* data;
    std::vector<size_t> shape;
    size_t size;
    std::vector<size_t> strides;

    size_t get_size() const;
    void init_strides();

    Tensor(){}
    Tensor(float* data, std::vector<size_t> shape);

    void copy_from(const Tensor& tensor);
    void copy_from(const float* new_data, size_t size_in_bytes);
    Tensor clone();

    Tensor at(std::initializer_list<size_t> idx);
    float max();

    Tensor reshape(std::vector<size_t> new_shape);

    void print();
};