#include <iostream>
#include "model/model.h"

int main(){
    std::shared_ptr<Parameters> params = std::make_shared<Parameters>();
    params->load_parameters("../model.bin");

    std::string text = "hello, how are you doing today?";
    auto res = params->tokenizer.encode(text);
    for (auto r : res){
        std::cout << r << std::endl;
    }

    Model model(params);

}
