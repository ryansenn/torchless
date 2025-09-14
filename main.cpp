#include <iostream>
#include "model/model.h"
#include "inference/math_ops.h"
#include "inference/inference.h"


int main(){
    Model model("../model.bin");

    std::string text = "hello how are you";
    std::vector<int> tokens = model.tokenizer->encode(text);

    InferenceState inferenceState(model);

    inferenceState.forward(tokens[0]);
}
