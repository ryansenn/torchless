#pragma once
#include <cassert>
#include <string>
#include <iostream>
#include "tensor.h"
#include "tokenizer.h"

struct Config {
    int vocab_size;
    int hidden_size;
    int num_hidden_layers;
};

struct Block {
    // Attention weights
    std::unique_ptr<Tensor> wq;
    std::unique_ptr<Tensor> wk;
    std::unique_ptr<Tensor> wv;
    std::unique_ptr<Tensor> wo;
};

// Load number of layers n
// Init n blocks
// Init attention weights in each block

struct Model {
    Config config;
    std::unique_ptr<Tensor> token_embedding_table;
    std::unique_ptr<Tokenizer> tokenizer; // tokenizer should probably not be a member of Model
    std::vector<Block> blocks;

    void load(std::string path);
    void load_metadata_entry(std::ifstream& f);
    void load_tensor_entry(std::ifstream& f);
};

struct InferenceState {
    Tensor x; // [hidden_size]

    InferenceState(Config& config) : x("x", {config.hidden_size, 0, 0, 0}){}
};