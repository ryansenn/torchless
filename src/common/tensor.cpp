#include "tensor.h"
#include <cassert>
#include <cstring>
#include <iostream>

size_t Tensor::get_numel() const {
    size_t s = 1;
    for (size_t d : shape) {
        assert(d > 0 && "dim=0 not supported");
        s *= d;
    }
    return s;
}

void Tensor::init_strides() {
    strides.assign(shape.size(), 1);
    if (shape.size() < 2) return;
    size_t stride = 1;
    for (int i = shape.size() - 2; i >= 0; --i) {
        stride *= shape[i+1];
        strides[i] = stride;
    }
}

Tensor::Tensor(float* data, const std::vector<size_t>& shape) : shape(shape), numel(get_numel()), data(data){
    init_strides();
}

Tensor::Tensor(Arena& arena, const std::vector<size_t>& shape) : shape(shape), numel(get_numel()), data(static_cast<float*>(arena.allocate(numel*type_size))){
    init_strides();
}

Tensor::Tensor(Arena& arena, const std::vector<float>& arr, const std::vector<size_t>& shape) : shape(shape), numel(get_numel()), data(static_cast<float*>(arena.allocate(numel*type_size))){
    init_strides();
    std::copy(arr.begin(), arr.end(), data);
}

void Tensor::copy_from(const Tensor& tensor) {
    std::memcpy(data, tensor.data, tensor.numel*type_size);
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
    for (int i=0; i<numel; i++){
        result = std::max(result, data[i]);
    }
    return result;
}

Tensor Tensor::reshape(std::vector<size_t> new_shape) {
    size_t new_numel = 1;
    for (auto d : new_shape) new_numel *= d;
    assert(new_numel <= numel && "Reshape size mismatch");
    return Tensor(data, new_shape);
}

void Tensor::print(){
    for (int i=0;i<numel;i++){
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}