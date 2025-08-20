#include <iostream>
#include "model/model.h"

int main(){
    Model model;
    model.load("../model.bin");

    std::string text = "hello how are you";
    std::vector<int> r = model.tokenizer->encode(text);

    for (auto i : r){
        std::cout << i << " ";
    }
    std::cout << std::endl;
}