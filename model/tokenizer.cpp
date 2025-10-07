#include "tokenizer.h"


uint64_t Tokenizer::pack(uint32_t left, uint32_t right){
    // Pack left
    uint64_t packed = left;
    // Shift to left and zero out 32 right-most bits
    packed = packed << 32;
    // Insert right on the right with OR
    packed = packed | right;

    return packed;
}



void Tokenizer::load(nlohmann::json tokenizer){
    id_to_token.resize(tokenizer["vocab"].size());

    // Load vocabulary tables
    for (auto& [key, value] : tokenizer["vocab"].items()){
        int id = value;
        id_to_token[id] = key;
        token_to_id[key] = id;
    }

    // Load BPE merge map
    int i = 0;
    for (auto& item : tokenizer["merges"]){
        std::string merge = item.get<std::string>();

        size_t s = merge.find(" ");
        std::string token1 = merge.substr(0, s);
        std::string token2 = merge.substr(s+1);

        uint32_t id1 = token_to_id[token1];
        uint32_t id2 = token_to_id[token2];

        uint64_t packed = pack(id1, id2);

        merge_to_rank[packed] = i;
        merge_to_id[packed] = token_to_id[token1 + token2];
        i++;
    }
}

// Get the merge pair with lowest rank in a list of tokens
// Returns UINT32_MAX when no merge is possible
uint64_t Tokenizer::get_lowest_pair(std::vector<uint32_t>& tokens){
    uint32_t lowest_rank = UINT64_MAX;
    uint64_t result = UINT64_MAX;

    for (int i=0;i<tokens.size()-1;i++){
        uint64_t packed = pack(tokens[i], tokens[i+1]);

        auto it = merge_to_rank.find(packed);
        if (it != merge_to_rank.end() && it->second < lowest_rank){
            lowest_rank = it->second;
            result = packed;
        }
    }

    return result;
}

// Merge all occurrences of the (left, right) token pair into merged token
std::vector<uint32_t> Tokenizer::merge(std::vector<uint32_t>& tokens, uint32_t left, uint32_t right, uint32_t merged){
    std::vector<uint32_t> merged_tokens;

    int i = 0;
    while (i < tokens.size()){
        if (tokens[i] == left && i + 1 < tokens.size() && tokens[i+1] == right){
            merged_tokens.push_back(merged);
            i += 2;
            continue;
        }

        merged_tokens.push_back(tokens[i]);
        i++;
    }

    return merged_tokens;
}

// Runs the BPE merge based tokenization
std::vector<uint32_t> Tokenizer::encode(std::string& text){
    // Transform into a list of token IDs
    std::vector<uint32_t> tokens;

    for (auto& c : text){
        std::string s(1, c);
        tokens.push_back(token_to_id[s]);
    }

    uint64_t packed = get_lowest_pair(tokens);

    while (packed != UINT64_MAX){

        uint32_t left  = static_cast<uint32_t>(packed >> 32);
        uint32_t right = static_cast<uint32_t>(packed & 0xFFFFFFFFu);

        tokens = merge(tokens, left, right, merge_to_id[packed]);
        packed = get_lowest_pair(tokens);
    }

    return tokens;
}

std::string Tokenizer::decode(std::vector<int>& tokens) {

}