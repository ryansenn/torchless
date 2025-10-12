#include "context.h"

int test_tokenizer_encode(){
    Model m = get_model();

    const std::string text = "The quick brown fox jumps over the lazy dog. Mistral models tokenize text using byte pair encoding, handling punctuation, emojis 😊, and multilingual words like déjà vu or 中文 with care.";
    std::vector<uint32_t> got = m.params->tokenizer.encode(text);

    const std::vector<uint32_t> expected = {
            1, 415, 2936, 9060, 285, 1142, 461, 10575, 754, 272,
            17898, 3914, 28723, 25200, 1650, 4994, 6029, 653, 2245, 1413,
            7500, 5964, 16087, 28725, 12852, 22195, 10223, 28725, 877, 6054,
            278, 28705, 30464, 28725, 304, 2531, 5708, 840, 3085, 737,
            28320, 20620, 442, 28705, 28991, 29019, 395, 1656, 28723
    };

    if (got.size() != expected.size()) {
        std::cout << "tokenizer encode: length mismatch, got "
                  << got.size() << " vs " << expected.size() << std::endl;
        return 1;
    }

    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] != expected[i]) {
            std::cout << "tokenizer encode: id mismatch at " << i
                      << ", got " << got[i] << " vs " << expected[i] << std::endl;
            return 1;
        }
    }

    return 0;
}

static RegisterTest load_weights("tokenizer encode",&test_tokenizer_encode);