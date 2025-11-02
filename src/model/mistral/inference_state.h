#pragma once
#include "../../common/tensor.h"
#include "../../loader/parameters.h"

// Holds tensor memory used during inference
struct InferenceState {
    Config config;
    Arena arena;
    Tensor hidden;

    InferenceState(Config& config) : config(config),
                                     arena(10 * 1024 * 1024), // 40MB, how much memory will be needed?
                                     hidden(arena, {50, config.hidden_size}) // Only 50 tokens at a time?
                                     {}


};