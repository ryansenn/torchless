#pragma once
#include "../../common/tensor.h"
#include "../../loader/parameters.h"

inline size_t MAX_SEQ_LEN = 50;

// Holds tensor memory used during inference
struct InferenceState {
    Config config;
    Arena arena;

    Tensor hidden_state; // [hidden_size]
    Tensor residual; // [hidden_size]
    size_t pos = 0;
    size_t seq_len = 0;

    Tensor inv_freq; // [head_dim / 2]
    Tensor cos; // [head_dim]
    Tensor sin; // [head_dim]

    Tensor q_state; // [n_heads, head_dim]
    Tensor k_state; // [n_kv_heads, head_dim]
    Tensor v_state; // [n_kv_heads, head_dim]

    Tensor k_cache; // [n_kv_heads, seq_len, head_dim]
    Tensor v_cache; // [n_kv_heads, seq_len, head_dim]

    Tensor scores; // [n_heads, seq_len]
    Tensor context; // [n_heads, head_dim]

    Tensor mlp_gate; // [intermediate_size]
    Tensor mlp_up; // [intermediate_size]

    Tensor logits; // [vocab_size]

    void push_kv(){
        for (size_t h=0;h<config.n_kv_heads;h++){
            k_cache.at({h, pos}).copy_from(k_state.at({h}));
            v_cache.at({h, pos}).copy_from(v_state.at({h}));
        }
    }

    InferenceState(Config& config) : config(config),
                                     arena(100 * 1024 * 1024), // 400MB, how much memory will be needed?
                                     hidden_state(arena, {config.hidden_size}), // Only 1 token at a time, pretty sure i will be having to rewrite this
                                     residual(arena, {config.hidden_size}),

                                     inv_freq(arena, {config.head_dim / 2}),
                                     cos(arena, {config.head_dim}),
                                     sin(arena, {config.head_dim}),

                                     q_state(arena, {config.n_heads, config.head_dim}),
                                     k_state(arena, {config.n_kv_heads, config.head_dim}),
                                     v_state(arena, {config.n_kv_heads, config.head_dim}),

                                     k_cache(arena, {config.n_kv_heads, MAX_SEQ_LEN, config.head_dim}),
                                     v_cache(arena, {config.n_kv_heads, MAX_SEQ_LEN, config.head_dim}),

                                     scores(arena, {config.n_heads, MAX_SEQ_LEN}),
                                     context(arena, {config.n_heads, config.head_dim}),

                                     mlp_gate(arena, {config.intermediate_size}),
                                     mlp_up(arena, {config.intermediate_size}),

                                     logits(arena, {config.vocab_size})

                                     {}
};