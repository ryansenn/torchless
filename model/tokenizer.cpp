#include "tokenizer.h"

std::shared_ptr<TrieNode> TrieNode::next_char(char c){
    children.emplace(c, std::make_shared<TrieNode>());
    return children[c];
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

std::vector<int> Tokenizer::encode(std::string text){
    return {};
}