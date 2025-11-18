#include "context.h"
#include <fstream>
#include <sstream>

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

void load_expected_values(){
    std::ifstream f("expected.txt");
    std::string line, name;

    while (std::getline(f, name)){
        if (name == "" || name[0] == '#') continue;
        std::getline(f, line);
        std::stringstream ss(line);
        float x;
        size_t i = 0;
        std::vector<float> data;
        while (ss >> x){
            data.push_back(x);
            i++;
        }
        expected.emplace(name, Tensor(arena, data, {i}));
    }
}

bool equals(float x, float y){
    float atol = 1e-2f;

    return std::fabs(x - y) < atol;
}

bool equals(const Tensor& x, const Tensor& y){
    if (x.size != y.size){
        return false;
    }

    for (int i=0; i<x.size; i++){
        if (!equals(x.data[i], y.data[i])){
            std::cout << "Mismatch at pos " << i
                      << ": expected " << y.data[i]
                      << ", got " << x.data[i] << std::endl;
            return false;
        }
    }

    return true;
}