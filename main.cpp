#include <iostream>
#include "model/model.h"
#include "math_ops.h"

int main(){
    Model model;
    model.load("../model.bin");

    std::string text = "hello how are you";
    std::vector<int> tokens = model.tokenizer->encode(text);

    InferenceState inferenceState(*model.config);
}

// Trying first minimal implementation of the inference flow
void forward(InferenceState& inferenceState, Model& model, int token){

}
