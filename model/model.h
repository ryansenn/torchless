#pragma once
#include <cassert>
#include <string>
#include <iostream>
#include "tensor.h"
#include "tokenizer.h"

struct Config {
    int vocab_size;
    int hidden_size;
};

struct Model {
    std::shared_ptr<Config> config;
    std::shared_ptr<Tensor> token_embedding_table;
    std::shared_ptr<Tokenizer> tokenizer; // tokenizer should probably not be a member of Model

    void load(std::string path);
    void load_metadata_entry(std::ifstream& f);
    void load_tensor_entry(std::ifstream& f);
};

struct InferenceState {

};