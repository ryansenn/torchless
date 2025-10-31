#pragma once
#include "tensor.h"

struct InferenceState {
    Tensor hidden;
    Tensor cos;
    Tensor sin;
    int position;
};