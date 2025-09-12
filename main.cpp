#include <iostream>
#include "model/model.h"
#include "math_ops.h"

void block_forward(InferenceState& inferenceState, Model& model, int b){
    // Get Q for the current token
    matmul(inferenceState.q, *model.blocks[b].wq, inferenceState.x);

    // Get K,V
    matmul(inferenceState.k, *model.blocks[b].wk, inferenceState.x);
    matmul(inferenceState.v, *model.blocks[b].wv, inferenceState.x);

    // Compute attention for each head individually
    for (int i=0; i<model.config.n_heads; i++){

    }

    // Write current K,V to cache
    inferenceState.k_cache[b]->at({inferenceState.pos}).copy_from(inferenceState.k);
    inferenceState.v_cache[b]->at({inferenceState.pos}).copy_from(inferenceState.v);
    inferenceState.pos++;
}

// Trying first minimal implementation of the inference flow
void forward(InferenceState& inferenceState, Model& model, int token){
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
