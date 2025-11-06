#pragma once
#include "../../common/tensor.h"
#include "../../loader/parameters.h"

inline size_t MAX_SEQ_LEN = 50;

// Holds tensor memory used during inference
struct InferenceState {
    Config config;
    Arena arena;
    Tensor hidden_state;
    size_t pos = 0;
    size_t seq_len = 0;

    Tensor inv_freq; // [head_dim / 2]
    Tensor cos; // [head_dim]
    Tensor sin; // [head_dim]

    InferenceState(Config& config) : config(config),
                                     arena(10 * 1024 * 1024), // 40MB, how much memory will be needed?
                                     hidden_state(arena, {config.hidden_size}), // Only 1 token at a time, pretty sure i will be having to rewrite this

                                     inv_freq(arena, {config.head_dim / 2}),
                                     cos(arena, {config.head_dim}),
                                     sin(arena, {config.head_dim})
                                     {}
};