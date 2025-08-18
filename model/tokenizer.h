#pragma once
#include "tensor.h"

struct TrieNode {
    std::unordered_map<char, std::shared_ptr<TrieNode>> children;
    int token_id = -1;
};

struct Tokenizer {
    std::vector<std::string> vocab;
    std::shared_ptr<TrieNode> trie;

    Tokenizer(std::shared_ptr<Tensor> raw_vocab);
    std::vector<int> encode(std::string text);
};