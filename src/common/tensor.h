#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <initializer_list>

struct Arena {
    size_t BUFFER_SIZE;
    char* buffer;
    size_t offset = 0;

    Arena(size_t BUFFER_SIZE) : BUFFER_SIZE(BUFFER_SIZE), buffer(new char[BUFFER_SIZE]) {}

    void* allocate(size_t size){
        assert(offset + size < BUFFER_SIZE && "Tensor allocator out of memory");
        char* result = buffer + offset;
        offset += size;

        return result;
    }

    ~Arena(){
        delete[] buffer;
    }
};

enum class Dtype {f32, i8};

struct Tensor {
    std::vector<size_t> shape;

    Dtype t;
    size_t numel;
    size_t type_size = 4;

    float* data;
    std::vector<size_t> strides;


    size_t get_numel() const;
    void init_strides();

    Tensor(float* data, const std::vector<size_t>& shape);
    Tensor(Arena& arena, const std::vector<size_t>& shape);
    Tensor(Arena& arena, const std::vector<float>& arr, const std::vector<size_t>& shape);

    void copy_from(const Tensor& tensor);

    Tensor at(std::initializer_list<size_t> idx);
    float max();

    Tensor reshape(std::vector<size_t> new_shape);

    void print();
};