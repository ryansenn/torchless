#include <unordered_map>
#include <string>
#include "../common/tensor.h"
#include "../tokenizer/tokenizer.h"
#include "../common/json.hpp"

#pragma once

struct Config {
    int hidden_size;
    int intermediate_size;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int max_seq_len;
    int head_dim;
    float rope_theta;
    float norm_eps;
};

struct Parameters {
    Config config;
    Tokenizer tokenizer;

    // Global weights (For Mistral, this is embeddings, final layernorm, lm_head)
    std::unordered_map<std::string, std::unique_ptr<Tensor>> global_weights;

    // Layer specific weights
    std::vector<std::unordered_map<std::string, std::unique_ptr<Tensor>>> layer_weights;

    static void* map_file(int fd);
    void load_tensor(std::unordered_map<std::string, std::unique_ptr<Tensor>>& m, char* p, const std::string& key, nlohmann::json& value);

    void load_config(nlohmann::json& header);
    void load_weights(char* p, nlohmann::json& header);
    void load_parameters(const std::string& path);
};