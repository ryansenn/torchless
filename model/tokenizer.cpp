#include "tokenizer.h"


Tokenizer::Tokenizer(Tensor* raw_vocab){
    char* p = reinterpret_cast<char*>(raw_vocab->data);
    int size = raw_vocab->shape[0];
    char* end = p+size;

    std::shared_ptr<TrieNode> n = trie;

    while (p < end){
        char* start = p;
        n = n->children[*p];

        while (*p != '\0' && p < end){
            p++;
            n = n->children[*p];
        }
        n->token_id = vocab.size();
        n = trie;
        vocab.push_back(std::string(start, p-start));
    }
}

std::vector<int> Tokenizer::encode(std::string text){

}