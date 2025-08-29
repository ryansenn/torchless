#pragma once
#include <cassert>
#include <string>
#include <iostream>
#include "tensor.h"
#include "tokenizer.h"
#include "json.hpp"

struct Config {
    int hidden_size;
    int intermediate_size;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int max_seq_len;
    float rope_theta;
    float norm_eps;
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

    Model(std::string path);
};

struct InferenceState {
    Tensor x; // [hidden_size]

    InferenceState(Config& config) : x("x", {config.hidden_size, 0, 0, 0}){}
};