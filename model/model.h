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
    // Input layer norm
    std::unique_ptr<Tensor> lm1;

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

    uint8_t* base_offset;

    Model(std::string path);
    std::unique_ptr<Tensor> load_tensor_by_key(const nlohmann::json& header, const std::string& key);
};

struct InferenceState {
    Tensor x; // [hidden_size]

    Tensor q;

    // KV cache
    // Each block stores Tensor of size [n_kv_heads, max_seq_len, head_dim]
    std::vector<std::unique_ptr<Tensor>> k_cache;
    std::vector<std::unique_ptr<Tensor>> v_cache;

    InferenceState(Config& config) :
        x("x", {config.hidden_size, 0, 0, 0}), q("q", {config.hidden_size, 0, 0, 0}){

        // Initialize empty KV cache
        int64_t head_dim = config.hidden_size / config.n_heads;
        std::vector<int64_t> shape = {config.n_kv_heads, config.max_seq_len, head_dim};

        for (int i=0; i<config.n_layers; i++){
            std::unique_ptr<Tensor> k = std::make_unique<Tensor>("k_cache_" + std::to_string(i), shape);
            std::unique_ptr<Tensor> v = std::make_unique<Tensor>("v_cache_" + std::to_string(i), shape);

            k_cache.push_back(std::move(k));
            v_cache.push_back(std::move(v));
        }
    }
};