#include "context.h"

int test_load_config(){
    Model m = get_model();

    if (m.params->config.hidden_size != 4096) {
        std::cout << "hidden_size mismatch: got " << m.params->config.hidden_size
                  << ", want 4096" << std::endl;
        return 1;
    }
    if (m.params->config.intermediate_size != 14336) {
        std::cout << "intermediate_size mismatch: got " << m.params->config.intermediate_size
                  << ", want 14336" << std::endl;
        return 1;
    }
    if (m.params->config.n_layers != 32) {
        std::cout << "n_layers mismatch: got " << m.params->config.n_layers
                  << ", want 32" << std::endl;
        return 1;
    }
    if (m.params->config.n_heads != 32) {
        std::cout << "n_heads mismatch: got " << m.params->config.n_heads
                  << ", want 32" << std::endl;
        return 1;
    }
    if (m.params->config.n_kv_heads != 8) {
        std::cout << "n_kv_heads mismatch: got " << m.params->config.n_kv_heads
                  << ", want 8" << std::endl;
        return 1;
    }
    if (m.params->config.vocab_size != 32000) {
        std::cout << "vocab_size mismatch: got " << m.params->config.vocab_size
                  << ", want 32000" << std::endl;
        return 1;
    }
    if (m.params->config.max_seq_len != 32768) {
        std::cout << "max_seq_len mismatch: got " << m.params->config.max_seq_len
                  << ", want 32768" << std::endl;
        return 1;
    }
    if (std::abs(m.params->config.rope_theta - 10000.0f) > 1e-3f) {
        std::cout << "rope_theta mismatch: got " << m.params->config.rope_theta
                  << ", want 10000.0" << std::endl;
        return 1;
    }
    if (std::abs(m.params->config.norm_eps - 1e-5f) > 1e-8f) {
        std::cout << "norm_eps mismatch: got " << m.params->config.norm_eps
                  << ", want 1e-5" << std::endl;
        return 1;
    }

    return 0;
}

static RegisterTest load_config("load config",&test_load_config);


/*
 * Test model weights for Mistral 7B
 */

struct Expected {
    std::string key;
    int layer; // -1 means global
    std::vector<int64_t> shape;
    float first;
    float last;
};

static const std::vector<Expected> expected_tensors = {
        // globals
        {"lm_head.weight",            -1, {32000, 4096}, -0.002593994140625f,      -7.82012939453125e-05f},
        {"model.embed_tokens.weight", -1, {32000, 4096}, -2.1864194925294548e-36f, -0.00250244140625f},
        {"model.norm.weight",         -1, {4096},         5.34375f,                 5.28125f},
        // first layer (layer 0)
        {"model.layers.0.input_layernorm.weight",          0, {4096},        -7.4803829193115234e-06f,  0.006591796875f},
        {"model.layers.0.post_attention_layernorm.weight", 0, {4096},         0.41796875f,              0.400390625f},
        {"model.layers.0.self_attn.q_proj.weight",         0, {4096, 4096},   5.3882598876953125e-05f, -0.000492095947265625f},
        {"model.layers.0.self_attn.k_proj.weight",         0, {1024, 4096},  -1.564621925354004e-06f,   0.000873565673828125f},
        {"model.layers.0.self_attn.v_proj.weight",         0, {1024, 4096},  -0.00041961669921875f,     0.00151824951171875f},
        {"model.layers.0.self_attn.o_proj.weight",         0, {4096, 4096},   0.000675201416015625f,   -0.00090789794921875f},
        {"model.layers.0.mlp.gate_proj.weight",            0, {14336, 4096}, -0.00421142578125f,       -0.003753662109375f},
        {"model.layers.0.mlp.up_proj.weight",              0, {14336, 4096}, -0.0001773834228515625f,   7.43865966796875e-05f},
        {"model.layers.0.mlp.down_proj.weight",            0, {4096, 14336}, -0.0026397705078125f,     -0.001953125f},

        // last layer (layer 31)
        {"model.layers.31.input_layernorm.weight",          31, {4096},        2.53125f,                 2.671875f},
        {"model.layers.31.post_attention_layernorm.weight", 31, {4096},        3.703125f,                3.71875f},
        {"model.layers.31.self_attn.q_proj.weight",         31, {4096, 4096},  0.000484466552734375f,   -0.00168609619140625f},
        {"model.layers.31.self_attn.k_proj.weight",         31, {1024, 4096},  0.001739501953125f,       0.0006866455078125f},
        {"model.layers.31.self_attn.v_proj.weight",         31, {1024, 4096},  0.00177001953125f,       -0.00049591064453125f},
        {"model.layers.31.self_attn.o_proj.weight",         31, {4096, 4096}, -0.0022430419921875f,     -0.001678466796875f},
        {"model.layers.31.mlp.gate_proj.weight",            31, {14336, 4096},  0.000270843505859375f,   0.00116729736328125f},
        {"model.layers.31.mlp.up_proj.weight",              31, {14336, 4096},  0.001495361328125f,       0.0013580322265625f},
        {"model.layers.31.mlp.down_proj.weight",            31, {4096, 14336},  0.00180816650390625f,     0.0024566650390625f},
};

int test_load_weights() {
    Model m = get_model();

    // Check we have the expected number of tensors loaded

    if (m.params->global_weights.size() != 3) {
        std::cerr << "Global tensor count mismatch: got "
                  << m.params->global_weights.size() << ", want 3\n";
        return 1;
    }

    if (m.params->layer_weights.size() != m.params->config.n_layers) {
        std::cerr << "Layer count mismatch: got "
                  << m.params->layer_weights.size() << ", want 32\n";
        return 1;
    }

    for (size_t i = 0; i < m.params->layer_weights.size(); ++i) {
        if (m.params->layer_weights[i].size() != 9) {
            std::cerr << "Layer " << i << " tensor count mismatch: got "
                      << m.params->layer_weights[i].size() << ", want 9\n";
            return 1;
        }
    }

    // Check some of the tensors have right shape, first value and last value

    for (const auto& e : expected_tensors) {
        const Tensor* t = nullptr;

        if (e.layer == -1) {
            auto it = m.params->global_weights.find(e.key);
            if (it == m.params->global_weights.end()) {
                std::cerr << "Missing global tensor: " << e.key << "\n";
                return 1;
            }
            t = it->second.get();
        } else {
            if (e.layer < 0 || e.layer >= (int)m.params->layer_weights.size()) {
                std::cerr << "Invalid layer index " << e.layer << " for " << e.key << "\n";
                return 1;
            }
            auto& lw = m.params->layer_weights[e.layer];
            auto it = lw.find(e.key);
            if (it == lw.end()) {
                std::cerr << "Missing tensor: " << e.key << "\n";
                return 1;
            }
            t = it->second.get();
        }

        const float* p = t->data;
        size_t n = t->get_size();
        if (!equals(p[0], e.first)) {
            std::cerr << e.key << " first val mismatch " << p[0] << " vs " << e.first << "\n";
            return 1;
        }
        if (!equals(p[n - 1], e.last)) {
            std::cerr << e.key << " last val mismatch " << p[n - 1] << " vs " << e.last << "\n";
            return 1;
        }
    }

    return 0;
}

static RegisterTest load_weights("load weights",&test_load_weights);
