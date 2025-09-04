#include "../model/model.h"
#include "registry.h"

Model& get_model() {
    static Model model("../model.bin");
    return model;
}

int test_model_load_metadata() {
    Model& m = get_model();

    if (m.config.hidden_size != 4096) {
        std::cout << "hidden_size mismatch: got " << m.config.hidden_size
                  << ", want 4096" << std::endl;
        return 1;
    }
    if (m.config.intermediate_size != 14336) {
        std::cout << "intermediate_size mismatch: got " << m.config.intermediate_size
                  << ", want 14336" << std::endl;
        return 1;
    }
    if (m.config.n_layers != 32) {
        std::cout << "n_layers mismatch: got " << m.config.n_layers
                  << ", want 32" << std::endl;
        return 1;
    }
    if (m.config.n_heads != 32) {
        std::cout << "n_heads mismatch: got " << m.config.n_heads
                  << ", want 32" << std::endl;
        return 1;
    }
    if (m.config.n_kv_heads != 8) {
        std::cout << "n_kv_heads mismatch: got " << m.config.n_kv_heads
                  << ", want 8" << std::endl;
        return 1;
    }
    if (m.config.vocab_size != 32000) {
        std::cout << "vocab_size mismatch: got " << m.config.vocab_size
                  << ", want 32000" << std::endl;
        return 1;
    }
    if (m.config.max_seq_len != 32768) {
        std::cout << "max_seq_len mismatch: got " << m.config.max_seq_len
                  << ", want 32768" << std::endl;
        return 1;
    }
    if (std::abs(m.config.rope_theta - 10000.0f) > 1e-3f) {
        std::cout << "rope_theta mismatch: got " << m.config.rope_theta
                  << ", want 10000.0" << std::endl;
        return 1;
    }
    if (std::abs(m.config.norm_eps - 1e-5f) > 1e-8f) {
        std::cout << "norm_eps mismatch: got " << m.config.norm_eps
                  << ", want 1e-5" << std::endl;
        return 1;
    }

    return 0;
}

static RegisterTest load_metadata("load metadata",&test_model_load_metadata);


int test_model_load_embedding() {
    Model& m = get_model();

    if (!m.token_embedding_table) {
        std::cout << "token_embedding_table is null" << std::endl;
        return 1;
    }

    float* data = m.token_embedding_table->data;
    float got = data[0];
    float expected = -2.18642e-36f;

    if (std::fabs(got - expected) > 1e-40f) {
        std::cout << "embedding[0] mismatch: got " << got
                  << ", want " << expected << std::endl;
        return 1;
    }

    return 0;
}

static RegisterTest load_embedding("load embedding", &test_model_load_embedding);

int test_tokenizer_vocab(){
    Model& m = get_model();

    std::vector<std::string>& vocab = m.tokenizer->vocab;

    if (vocab.size() != 32000) {
        std::cout << "vocab size mismatch: got " << vocab.size()
                  << ", want 32000" << std::endl;
        return 1;
    }

    if (vocab[31855] != "拥") {
        std::cout << "vocab[31855] mismatch: got " << vocab[31855]
                  << ", want 拥" << std::endl;
        return 1;
    }

    if (vocab[1098] != "ular") {
        std::cout << "vocab[1098] mismatch: got " << vocab[1098]
                  << ", want ular" << std::endl;
        return 1;
    }

    if (vocab[0] != "<unk>") {
        std::cout << "vocab[0] mismatch: got " << vocab[0]
                  << ", want <unk>" << std::endl;
        return 1;
    }

    return 0;
}

static RegisterTest tokenizer_vocab("tokenizer vocab", &test_tokenizer_vocab);

int test_tokenizer_encode_basic(){
    Model& m = get_model();
    if (!m.tokenizer) {
        std::cout << "tokenizer is null" << std::endl;
        return 1;
    }

    std::string text = "hello how are you";
    std::vector<int> want = {21558, 910, 460, 368};

    std::vector<int> got = m.tokenizer->encode(text);

    if (got.size() != want.size()) {
        std::cout << "encode size mismatch: got " << got.size()
                  << ", want " << want.size() << std::endl;
        return 1;
    }
    for (size_t i = 0; i < want.size(); ++i) {
        if (got[i] != want[i]) {
            std::cout << "encode mismatch at " << i << ": got " << got[i]
                      << ", want " << want[i] << std::endl;
            return 1;
        }
    }
    return 0;
}

static RegisterTest tokenizer_encode_basic("tokenizer encode basic", &test_tokenizer_encode_basic);