#include "setup/context.h"

int test_layer() {
    std::shared_ptr<Parameters> params = get_params();
    infer.pos = 0;

    Layer layer(params->layer_weights[0]);

    // Test over sequence of 3 tokens
    for (int i=1;i<4;i++) {
        infer.hidden_state.copy_from(expected.at("layer_h" + std::to_string(i)));
        layer.forward(infer);

        if (!equals(infer.hidden_state, expected.at("layer_o" + std::to_string(i)))){
            std::cout << "Layer mismatch at token " + std::to_string(i) << std::endl;
            return 1;
        }
    }

    return 0;
}

RegisterTest layer_reg("test layer", &test_layer);

int test_attention() {
    std::shared_ptr<Parameters> params = get_params();
    infer.pos = 0;

    auto& w = params->layer_weights[0];
    Attention attn(w.at("self_attn.q_proj.weight"), w.at("self_attn.k_proj.weight"), w.at("self_attn.v_proj.weight"), w.at("self_attn.o_proj.weight"));

    // Test over sequence of 3 tokens
    for (int i=1;i<4;i++){
        infer.hidden_state.copy_from(expected.at("attn_h" + std::to_string(i)));
        attn.forward(infer);

        if (!equals(infer.q_state, expected.at("attn_q" + std::to_string(i)))){
            std::cout << "Attention q mismatch at token " + std::to_string(i) << std::endl;
            return 1;
        }

        if (!equals(infer.k_state, expected.at("attn_k" + std::to_string(i)))){
            std::cout << "Attention k mismatch at token "  + std::to_string(i)  << std::endl;
            return 1;
        }

        if (!equals(infer.v_state, expected.at("attn_v" + std::to_string(i)))){
            std::cout << "Attention v mismatch at token "  + std::to_string(i)  << std::endl;
            return 1;
        }

        /*
        if (!equals(infer.scores.reshape({infer.scores.shape[0], infer.pos+1}), expected.at("attn_w" + std::to_string(i)))){
            std::cout << "Attention weights mismatch at token " + std::to_string(i)  << std::endl;
            return 1;
        }
        */

        if (!equals(infer.hidden_state, expected.at("attn_o" + std::to_string(i)))){
            std::cout << "Attention result mismatch at token " + std::to_string(i)  << std::endl;
            return 1;
        }
        infer.pos++;
    }

    return 0;
}

RegisterTest attention_reg("test attention", &test_attention);

int test_mlp(){
    std::shared_ptr<Parameters> params = get_params();
    auto w = params->layer_weights[0];

    MLP mlp(w.at("mlp.down_proj.weight"),
            w.at("mlp.gate_proj.weight"),
            w.at("mlp.up_proj.weight"));

    infer.hidden_state.copy_from(expected.at("mlp_h"));

    mlp.forward(infer);

    if (!equals(infer.hidden_state, expected.at("mlp_output"))){
        std::cout << "MLP feedforward result mismatch" << std::endl;
        return 1;
    }

    return 0;
}

RegisterTest mlp_feedforward_reg("test attention feedforward mlp", &test_mlp);

int test_kv_cache() {
    infer.pos = 5;

    Tensor dummy(arena, infer.k_state.shape);
    for(int i=0;i<dummy.size;i++){
        dummy.data[i] = float(i);
    }

    infer.k_state.copy_from(dummy);
    infer.v_state.copy_from(dummy);

    infer.push_kv();

    for (size_t h=0; h<infer.config.n_kv_heads; h++){
        if (!equals(infer.k_cache.at({h, infer.pos}), dummy.at({h}))){
            std::cout << "KV Cache push k mismatch" << std::endl;
            return 1;
        }

        if (!equals(infer.v_cache.at({h, infer.pos}), dummy.at({h}))){
            std::cout << "KV Cache push v mismatch" << std::endl;
            return 1;
        }
    }

    return 0;
}

RegisterTest kv_cache_reg("test kv cache", &test_kv_cache);

int test_embedding() {
    std::shared_ptr<Parameters> params = get_params();

    Embedding emb(params->global_weights.at("model.embed_tokens.weight"));

    size_t token_id = 0;
    emb.forward(infer, token_id);

    Tensor emb1 = infer.hidden_state;

    if (!equals(emb1.data[0], -2.1864e-36f)) {
        std::cout << "emb1[0][0] mismatch" << std::endl;
        return 1;
    }
    if (!equals(emb1.data[4095], -6.3947e-36f)) {
        std::cout << "emb1[0][-1] mismatch" << std::endl;
        return 1;
    }

    token_id = 31999;
    emb.forward(infer, token_id);
    Tensor emb2 = infer.hidden_state;

    if (!equals(emb2.data[0], -0.0040f)) {
        std::cout << "emb2[-1][0] mismatch" << std::endl;
        return 1;
    }
    if (!equals(emb2.data[4095], -0.0025f)) {
        std::cout << "emb2[-1][-1] mismatch" << std::endl;
        return 1;
    }

    return 0;
}

RegisterTest embedding_reg("test embedding", &test_embedding);

int test_rotary_embedding_inv_freq(){
    std::shared_ptr<Parameters> params = get_params();

    RotaryEmbedding::init_freq(infer, params->config);

    if (!equals(expected.at("inv_freq"), infer.inv_freq)){
        std::cout << "Rotary embedding inv freq mismatch" << std::endl;
        return 1;
    }

    return 0;
}

RegisterTest rotary_embedding_inv_freq_reg("test rotary embedding inv freq", &test_rotary_embedding_inv_freq);

int test_rotary_embedding(){
    std::shared_ptr<Parameters> params = get_params();

    RotaryEmbedding::init_freq(infer, params->config);

    infer.pos = 0;
    RotaryEmbedding::forward(infer);

    if (!equals(infer.cos, expected.at("cos0"))){
        std::cout << "rotary embedding cos mismatch pos 0" << std::endl;
        return 1;
    }

    if (!equals(infer.sin, expected.at("sin0"))){
        std::cout << "rotary embedding sin mismatch pos 0" << std::endl;
        return 1;
    }

    infer.pos = 3;
    RotaryEmbedding::forward(infer);

    if (!equals(infer.cos, expected.at("cos3"))){
        std::cout << "rotary embedding cos mismatch pos 3" << std::endl;
        return 1;
    }

    if (!equals(infer.sin, expected.at("sin3"))){
        std::cout << "rotary embedding sin mismatch pos 3" << std::endl;
        return 1;
    }

    return 0;
}

RegisterTest rotary_embeddingg("test rotary embedding", &test_rotary_embedding);



int test_rmsnorm() {
    infer.hidden_state.copy_from(expected.at("norm_x"));
    Tensor g = expected.at("norm_g");
    Tensor y = expected.at("norm_y");

    RMSNorm rms(g);
    rms.forward(infer);

    if (!equals(infer.hidden_state, y)) {
        std::cout << "RMSNorm mismatch" << std::endl;
        return 1;
    }

    return 0;
}

RegisterTest rmsnorm_reg("test rmsnorm", &test_rmsnorm);
