#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <initializer_list>

struct Tensor {
    float* data; // need to free this at some point
    std::vector<int64_t> shape;
    size_t size;
    std::vector<int64_t> strides;

    size_t get_size() const;
    void init_strides();

    Tensor(){}
    Tensor(float* data, std::vector<int64_t> shape);
    Tensor(std::vector<int64_t> shape);
    Tensor(std::vector<float> arr, std::vector<int64_t> shape);

    void copy_from(const Tensor& tensor);
    void copy_from(const float* new_data, size_t size_in_bytes);

    Tensor at(std::initializer_list<int64_t> idx);

    Tensor reshape(std::vector<int64_t> new_shape);

    Tensor slice1d(int start, int len);

    void check_shape(const std::vector<int64_t>& expected_shape) const;
};