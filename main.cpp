#include <iostream>
#include "model/model.h"
#include "inference/math_ops.h"
#include "inference/inference.h"

void block_forward(InferenceState& inferenceState, int b){
    Model& model = inferenceState.model;

    // Get Q for the current token
    matmul(inferenceState.q, *model.blocks[b].wq, inferenceState.x);


    // Compute attention for each head individually
    for (int i=0; i<model.config.n_heads; i++){

    }

    // Write current K,V to cache
    inferenceState.push_kv(b);
}

// Trying first minimal implementation of the inference flow
void forward(InferenceState& inferenceState, int token){
    float* embedding = &model.token_embedding_table->data[token * model.config.hidden_size];
    inferenceState.x.copy_from(embedding, model.config.hidden_size);

    rmsnorm(inferenceState.x, inferenceState.x, *model.blocks[0].lm1, model.config.hidden_size);

    // Forward for each block
    for (int i=0; i<model.config.n_layers; i++){
        block_forward(inferenceState, model, i);
    }
}

int main(){
    Model model("../model.bin");

    std::string text = "hello how are you";
    std::vector<int> tokens = model.tokenizer->encode(text);

    InferenceState inferenceState(model.config);

    forward(inferenceState, model, tokens[0]);

}
