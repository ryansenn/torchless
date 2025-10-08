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

        // Bytes in the vocab JSON are written as strings representation "<0x01>"
        // Im converting them to an actual byte so that I can look them up directly during encoding
        if (key.substr(0, 3) == "<0x"){
            std::string hexa = key.substr(3, 2);
            unsigned char byte = std::stoi(hexa, nullptr, 16);
            std::string byte_key = std::string(1, byte);

            int id = value;
            id_to_token[id] = byte_key;
            token_to_id[byte_key] = id;

            continue;
        }

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
// Returns UINT64_MAX when no merge is possible
uint64_t Tokenizer::get_lowest_pair(std::vector<uint32_t>& tokens){
    uint32_t lowest_rank = UINT32_MAX;
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

// For mistral, use Metaspace pre-tokenization, replace spaces with '▁' and prepend
std::string Tokenizer::pre_tokenize_mistral(std::string& text){
    std::string out = "▁";

    for (auto c : text){
        if (c == ' '){
            out += "▁";
            continue;
        }
        out += c;
    }
    return out;
}

// Runs the BPE merge based tokenization
std::vector<uint32_t> Tokenizer::encode(std::string text){
    // If want to support other tokenizers config we would support other stages that are not needed in Mistral

    // Normalization (not implemented for Mistral)

    // Pre-tokenization
    text = pre_tokenize_mistral(text);

    // Now run the BPE Algorithm

    // Transform into a list of token IDs
    std::vector<uint32_t> tokens;

    size_t i = 0;
    while (i < text.size()) {
        // If we have a valid multiple byte UTF-8 encoding, we want to merge them into one string
        // I should probably use a library to handle raw UTF-8 bytes but couldnt find
        // The first byte leading 1s indicate the number of byte to read

        unsigned char b0 = static_cast<unsigned char>(text[i]);
        size_t len = 1;

        if      ((b0 & 0x80) == 0x00) len = 1;          // 0xxxxxxx
        else if ((b0 & 0xE0) == 0xC0) len = 2;          // 110xxxxx
        else if ((b0 & 0xF0) == 0xE0) len = 3;          // 1110xxxx
        else if ((b0 & 0xF8) == 0xF0) len = 4;          // 11110xxx

        std::string s = text.substr(i, len);
        tokens.push_back(token_to_id[s]);
        i += len;
    }

    uint64_t packed = get_lowest_pair(tokens);

    // While we still find a valid pair to merge
    while (packed != UINT64_MAX){
        uint32_t left  = static_cast<uint32_t>(packed >> 32);
        uint32_t right = static_cast<uint32_t>(packed & 0xFFFFFFFFu);

        // Perform the merge on tokens
        tokens = merge(tokens, left, right, merge_to_id[packed]);
        packed = get_lowest_pair(tokens);
    }

    // Mistral post-processing, BOS token
    tokens.insert(tokens.begin(), token_to_id.at("<s>"));

    return tokens;
}

std::string Tokenizer::decode(std::vector<int>& tokens) {
    return "hello!";
}