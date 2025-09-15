#include "context.h"

std::vector<TestCase> tests;

Model& get_model(){
    static Model model("../model.bin");
    return model;
}

bool equals(float x, float y){
    float atol = 1e-5f;

    return std::fabs(x - y) < atol;
}