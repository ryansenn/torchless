#include "context.h"

std::vector<TestCase> tests;

Model& get_model(){
    static Model model("../model.bin");
    return model;
}