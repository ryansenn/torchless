#include "setup/context.h"


int test_embedding() {
    std::shared_ptr<Parameters> params = get_params();

    Embedding emb(*params->global_weights["model.embed_tokens.weight"]);

    std::vector<size_t> idx{0, 31999};
    emb.forward(infer, idx);

    Tensor emb1 = infer.hidden_state.at({0});
    Tensor emb2 = infer.hidden_state.at({1});

    if (!equals(emb1.data[0], -2.1864e-36f)) {
        std::cout << "emb1[0][0] mismatch" << std::endl;
        return 1;
    }
    if (!equals(emb1.data[4095], -6.3947e-36f)) {
        std::cout << "emb1[0][-1] mismatch" << std::endl;
        return 1;
    }
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


/*
int test_rmsnorm() {
    Tensor x(arena, {2.4f, 7.8f, 1.1f, 5.3f, 9.0f, 4.7f, 6.2f, 3.8f, 8.5f, 0.6f}, {10});
    Tensor g(arena, {0.8f, 1.2f, 1.0f, 0.9f, 1.1f, 1.3f, 0.7f, 1.0f, 1.2f, 0.95f}, {10});

    RMSNorm rms(g);
    Tensor out = rms.forward(x);

    Tensor expected(arena, {0.3371f, 1.6432f, 0.1931f, 0.8374f, 1.7380f, 1.0726f, 0.7619f, 0.6671f, 1.7906f, 0.1001f}, {10});

    if (!equals(out, expected)) {
        std::cout << "RMSNorm mismatch" << std::endl;
        return 1;
    }

    return 0;
}

RegisterTest rmsnorm_reg("test rmsnorm", &test_rmsnorm);
*/