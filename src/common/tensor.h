#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <initializer_list>

// TODO: Either make Tensor fully own its memory or be a pure view.
// TODO: Make it memory safe
struct Tensor {
    float* data; // need to free this at some point
    std::vector<size_t> shape;
    size_t size;
    std::vector<size_t> strides;

    size_t get_size() const;
    void init_strides();

    Tensor(){}
    Tensor(float* data, std::vector<size_t> shape);
    Tensor(std::vector<size_t> shape);
    Tensor(std::vector<float> arr, std::vector<size_t> shape);

    void copy_from(const Tensor& tensor);
    void copy_from(const float* new_data, size_t size_in_bytes);
    Tensor clone();

    Tensor at(std::initializer_list<size_t> idx);
    float max();

    Tensor reshape(std::vector<size_t> new_shape);

    void check_shape(const std::vector<size_t>& expected_shape) const;

    void print();
};