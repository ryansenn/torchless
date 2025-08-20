#include <iostream>
#include "model/model.h"

int main(){
    Model model;
    model.load("../model.bin");

    std::cout << model.tokenizer->vocab[433] << std::endl;

}