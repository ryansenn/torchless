#pragma once
#include "tensor.h"
#include "json.hpp"

struct Tokenizer {
    // Vocab lookup tables
    // Store the vocab to go back and forth Token <-> ID
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, uint32_t> token_to_id;

    // BPE Merge table
    // Maps a merged token pair to its merge rank (Vocab token ID)
    // Each key packs two 32-bit token IDs (left and right) into one 64-bit integer
    std::unordered_map<uint64_t, uint32_t> merge_to_rank;

    // Maps a packed (left,right) token pair to its merged token ID to avoid recomputing merges during encoding
    std::unordered_map<uint64_t, uint32_t> merge_to_id;

    uint32_t get_id(const std::string& token) const;

    uint64_t pack(uint32_t left, uint32_t right) const;
    uint64_t get_lowest_pair(std::vector<uint32_t>& tokens) const;
    std::vector<uint32_t> merge(std::vector<uint32_t>& tokens, uint32_t left, uint32_t right, uint32_t merged) const;

    void load(nlohmann::json tokenizer);

    std::string pre_tokenize_mistral(std::string& text) const;
    std::vector<uint32_t> encode(std::string text) const;
    std::string decode(std::vector<int>& tokens) const;
};