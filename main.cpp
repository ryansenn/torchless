#include <iostream>
#include "model/model.h"

int main(){
    std::shared_ptr<Parameters> params = std::make_shared<Parameters>();
    params->load_parameters("../model.bin");

    Model model(params);
}
