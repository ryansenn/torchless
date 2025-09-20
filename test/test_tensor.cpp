#include "context.h"

int test_tensor_index(){
    float data[4][4][2] = {
            { {0,1}, {2,3}, {4,5}, {6,7} },
            { {8,9}, {10,11}, {12,13}, {14,15} },
            { {16,17}, {18,19}, {20,21}, {22,23} },
            { {24,25}, {26,27}, {28,29}, {30,31} }
    };
    std::vector<int64_t> shape = {4, 4, 2};
    Tensor t("test", reinterpret_cast<float*>(data), shape);

    // Scalar access: t[1,2,0] == 12
    {
        Tensor v = t.at({1,2,0});
        if (!v.shape.empty() || *(v.data) != 12.f) {
            std::cout << "tensor_index scalar failed\n";
            return 1;
        }
    }

    // 1D slice: t[3,0] -> shape {2}, values {24,25}
    {
        Tensor v = t.at({3,0});
        if (v.shape.size() != 1 || v.shape[0] != 2) {
            std::cout << "tensor_index 1D shape failed\n";
            return 1;
        }
        if (v.data[0] != 24.f || v.data[1] != 25.f) {
            std::cout << "tensor_index 1D values failed\n";
            return 1;
        }
    }

    // 2D slice: t[2] -> shape {4,2}; check first and last elements (16, 23)
    {
        Tensor v = t.at({2});
        if (v.shape.size() != 2 || v.shape[0] != 4 || v.shape[1] != 2) {
            std::cout << "tensor_index 2D shape failed\n";
            return 1;
        }
        if (*(v.at({0,0}).data) != 16.f || *(v.at({3,1}).data) != 23.f) {
            std::cout << "tensor_index 2D values failed\n";
            return 1;
        }
    }

    return 0;
}

static RegisterTest reg_tensor_index("tensor_index", &test_tensor_index);

int test_tensor_reshape() {
    // Create a 2x3x4 tensor filled with values 0..23
    float data[24];
    for (int i = 0; i < 24; i++) data[i] = static_cast<float>(i);

    std::vector<int64_t> shape = {2, 3, 4};
    Tensor t("reshape_test", data, shape);

    // Reshape to 4x6
    Tensor r = t.reshape({4, 6});

    // Check shape
    if (r.shape.size() != 2 || r.shape[0] != 4 || r.shape[1] != 6) {
        std::cout << "tensor_reshape shape failed\n";
        return 1;
    }

    // Check that data is shared and values line up
    for (int i = 0; i < 24; i++) {
        if (r.data[i] != static_cast<float>(i)) {
            std::cout << "tensor_reshape data mismatch at " << i << "\n";
            return 1;
        }
    }

    // Now reshape back to original and check again
    Tensor r2 = r.reshape({2, 3, 4});
    if (r2.shape.size() != 3 || r2.shape[0] != 2 || r2.shape[1] != 3 || r2.shape[2] != 4) {
        std::cout << "tensor_reshape roundtrip shape failed\n";
        return 1;
    }
    if (r2.data[5] != 5.f || r2.data[23] != 23.f) {
        std::cout << "tensor_reshape roundtrip values failed\n";
        return 1;
    }

    return 0;
}

static RegisterTest reg_tensor_reshape("tensor_reshape", &test_tensor_reshape);