#include <iostream>
#include "model/model.h"

int main(){
    Model model;
    model.load("../model.bin");

    std::string text = "hello how are you";
    std::vector<int> tokens = model.tokenizer->encode(text);

    InferenceState inferenceState(*model.config);
}

