#include <iostream>
#include "model/model.h"

int main(){
    Model model;
    model.load("model.bin");

    std::cout << model.config.hidden_size << " " << model.config.vocab_size << std::endl;
}