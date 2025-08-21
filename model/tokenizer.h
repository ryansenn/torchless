#pragma once
#include "tensor.h"

struct TrieNode {
    std::unordered_map<char, std::shared_ptr<TrieNode>> children;
    int token_id = -1;

    std::shared_ptr<TrieNode> next_char(char c);
    bool contains(char c);
};

struct Tokenizer {
    std::vector<std::string> vocab;
    std::shared_ptr<TrieNode> trie;

    Tokenizer(std::shared_ptr<Tensor> raw_vocab);
    std::vector<int> encode(std::string& text);
    std::string decode(std::vector<int>& tokens);
};