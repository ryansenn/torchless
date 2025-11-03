#include "modules.h"


void Embedding::forward(InferenceState& infer, const std::vector<size_t>& ids){
    for (size_t i=0;i<ids.size();i++){
        infer.hidden.at({i}).copy_from(table.at({i}));
    }
}