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
    Config config;
    Tensor* token_embedding_table;

    // To Refactor: Tokenizer should probably not be a member of Model
    Tokenizer* tok;

    void load(std::string path);
};

struct InferenceState {

};