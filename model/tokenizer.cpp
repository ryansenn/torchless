#include "tokenizer.h"

std::shared_ptr<TrieNode> TrieNode::next_char(char c){
    children.emplace(c, std::make_shared<TrieNode>());
    return children[c];
}

bool TrieNode::contains(char c){
    return children.find(c) != children.end();
}

Tokenizer::Tokenizer(std::shared_ptr<Tensor> raw_vocab){
    trie = std::make_shared<TrieNode>();

    char* p = reinterpret_cast<char*>(raw_vocab->data);
    int size = raw_vocab->shape[0];
    char* end = p+size;

    std::shared_ptr<TrieNode> n = trie;

    while (p < end){
        char* start = p;

        n = n->next_char(*p);
        p++;

        while (*p != '\0' && p < end){
            n = n->next_char(*p);
            p++;
        }
        n->token_id = vocab.size();
        n = trie;
        std::string word = std::string(start, p-start);
        vocab.push_back(word);
        p++;
    }
}

std::vector<int> Tokenizer::encode(std::string& text){
    std::vector<int> result;

    std::shared_ptr<TrieNode> n = trie;
    int i = 0;

    int curr_id = -1;

    while (i < text.size()){
        if (n->contains(text[i])){
            n = n->next_char(text[i]);
            if (n->token_id > -1){
                curr_id = n->token_id;
            }
            i++;
        }
        else {
            result.push_back(curr_id);
            n = trie;
            curr_id = -1;
        }
    }

    if (curr_id > -1){
        result.push_back(curr_id);
    }

    return result;
}

// i think need to figure out how to strip extra space and deal with special characters
std::string Tokenizer::decode(std::vector<int>& tokens) {
    std::string result;
    for (int t : tokens) {
            result += vocab[t];
    }
    return result;
}