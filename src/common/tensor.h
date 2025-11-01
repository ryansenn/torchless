#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <initializer_list>

struct Arena {
    size_t BUFFER_SIZE;
    float* buffer;
    size_t offset = 0;

    Arena(size_t BUFFER_SIZE) : BUFFER_SIZE(BUFFER_SIZE), buffer(new float[BUFFER_SIZE]) {}

    float* allocate(size_t size){
        assert(offset + size < BUFFER_SIZE && "Tensor allocator out of memory");
        float* result = buffer + offset;
        offset += size;

        return result;
    }

    ~Arena(){
        delete[] buffer;
    }
};

struct Tensor {
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    size_t size;
    float* data;

    size_t get_size() const;
    void init_strides();

    Tensor(float* data, const std::vector<size_t>& shape);
    Tensor(Arena& arena, const std::vector<size_t>& shape);
    Tensor(Arena& arena, const std::vector<float>& arr, std::vector<size_t>& shape);

    void copy_from(const Tensor& tensor);

    Tensor at(std::initializer_list<size_t> idx);
    float max();

    Tensor reshape(std::vector<size_t> new_shape);

    void print();
};