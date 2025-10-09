#include "context.h"

std::vector<TestCase> tests;

Model& get_model(){
    static bool init = false;

    static std::shared_ptr<Parameters> params = std::make_shared<Parameters>();
    static Model model(params);

    if (init){
        return model;
    }

    params->load_parameters("../model.bin");
    init = true;

    return model;
}

bool equals(float x, float y){
    float atol = 1e-5f;

    return std::fabs(x - y) < atol;
}