#include "context.h"

std::vector<TestCase> tests;

std::shared_ptr<Parameters> get_params(){
    static bool init = false;

    static std::shared_ptr<Parameters> params = std::make_shared<Parameters>();

    if (init){
        return params;
    }

    params->load_parameters("../model.bin");
    init = true;

    return params;
}

bool equals(float x, float y){
    float atol = 1e-2f;

    return std::fabs(x - y) < atol;
}

bool equals(Tensor& x, Tensor& y){
    if (x.shape != y.shape){
        return false;
    }

    for (int i=0; i<x.size; i++){
        if (!equals(x.data[i], y.data[i])){
            return false;
        }
    }

    return true;
}