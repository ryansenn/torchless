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
    Tensor cos; // [50, head_dim]
    Tensor sin; // [50, head_dim]

    InferenceState(Config& config) : config(config),
                                     arena(10 * 1024 * 1024), // 40MB, how much memory will be needed?
                                     hidden_state(arena, {MAX_SEQ_LEN, config.hidden_size}), // Only 50 tokens at a time?

                                     inv_freq(arena, {config.head_dim / 2}),
                                     cos(arena, {MAX_SEQ_LEN, config.head_dim}),
                                     sin(arena, {MAX_SEQ_LEN, config.head_dim})
                                     {}
};