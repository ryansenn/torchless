#include <iostream>
#include "model/model.h"
#include "math_ops.h"

// Trying first minimal implementation of the inference flow
void forward(InferenceState& inferenceState, Model& model, int token){
    float* embedding = &model.token_embedding_table->data[token * model.config.hidden_size];
    inferenceState.x.copy_from(embedding, model.config.hidden_size);


}

int main(){
    Model model("../model.bin");

    std::string text = "hello how are you";
    std::vector<int> tokens = model.tokenizer->encode(text);

    InferenceState inferenceState(model.config);

    forward(inferenceState, model, tokens[0]);

}
