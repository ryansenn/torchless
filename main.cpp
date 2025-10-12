#include <iostream>
#include "src/loader/parameters.h"

int main(){
    std::shared_ptr<Parameters> params = std::make_shared<Parameters>();
    params->load_parameters("../model.bin");

}
