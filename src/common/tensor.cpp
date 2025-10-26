#include "tensor.h"
#include <cassert>
#include <cstring>
#include <iostream>

size_t Tensor::get_size() const {
    size_t s = 1;
    for (auto d : shape) {
        assert(d > 0 && "dim=0 not supported");
        s *= static_cast<size_t>(d);
    }
    return s;
}

void Tensor::init_strides() {
    strides.assign(shape.size(), 1);
    if (shape.size() < 2) return;
    int64_t stride = 1;
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        stride *= shape[i+1];
        strides[i] = stride;
    }
}

Tensor::Tensor(float* data, std::vector<size_t> shape)
        : data(data), shape(std::move(shape)) {
    size = get_size();
    init_strides();
}

Tensor::Tensor(std::vector<size_t> shape)
        : shape(std::move(shape)) {
    size = get_size();
    data = new float[size]; // need to free this at some point
    init_strides();
}

Tensor::Tensor(std::vector<float> arr, std::vector<size_t> shape) : shape(std::move(shape)) {
    size = get_size();
    data = new float[size]; // need to free this at some point
    init_strides();

    assert(arr.size() == size && "Tensor init mismatch between data and shape");

    for(int i=0;i<arr.size();i++){
        data[i] = arr[i];
    }
}

void Tensor::copy_from(const Tensor& tensor) {
    std::memcpy(data, tensor.data, tensor.get_size() * sizeof(float));
}

Tensor Tensor::clone(){
    Tensor out(shape);
    out.copy_from(*this);
    return out;
}

Tensor Tensor::at(std::initializer_list<size_t> idx) {
    assert(idx.size() <= shape.size() && "Too many indices for tensor");
    float* new_data = data;

    int i = 0;
    for (auto v : idx) {
        assert(v < shape[i] && "Index out of range");
        new_data += strides[i] * v;
        i++;
    }

    std::vector<size_t> new_shape(shape.begin() + i, shape.end());
    return Tensor(new_data, new_shape);
}

float Tensor::max(){
    float result = data[0];

    for (int i=0; i<size; i++){
        result = std::max(result, data[i]);
    }

    return result;
}

Tensor Tensor::reshape(std::vector<size_t> new_shape) {
    size_t new_size = 1;
    for (auto d : new_shape) new_size *= d;
    assert(new_size == size && "Reshape size mismatch");
    return Tensor(data, new_shape);
}

void Tensor::check_shape(const std::vector<size_t>& expected_shape) const {
    if (shape != expected_shape) {
        std::cerr << "FATAL: shape mismatch" << std::endl;

        std::cerr << "Expected: [";
        for (size_t i = 0; i < expected_shape.size(); ++i) {
            std::cerr << expected_shape[i] << (i+1 < expected_shape.size() ? ", " : "");
        }
        std::cerr << "]\nGot: [";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cerr << shape[i] << (i+1 < shape.size() ? ", " : "");
        }
        std::cerr << "]" << std::endl;

        assert(false);
    }
}

void Tensor::print(){
    for (int i=0;i<size;i++){
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}