#include <unordered_map>
#include <string>
#include "../common/tensor.h"
#include "../tokenizer/tokenizer.h"
#include "../common/json.hpp"

#pragma once

struct Config {
    size_t hidden_size;
    size_t intermediate_size;
    size_t n_layers;
    size_t n_heads;
    size_t n_kv_heads;
    size_t vocab_size;
    size_t max_seq_len;
    size_t head_dim;
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