#include <iostream>
#include "src/loader/parameters.h"
#include "src/model/mistral/modules.h"

int main(){
    std::shared_ptr<Parameters> params = std::make_shared<Parameters>();
    params->load_parameters("../model.bin");
    InferenceState infer(params->config);

    Model model(params);
    model.forward(infer, 1);

    infer.hidden_state.print();

    return 0;
}
