#include <unordered_map>
#include <string>
#include "tensor.h"
#include "tokenizer.h"
#include "json.hpp"

struct Config {
    int hidden_size;
    int intermediate_size;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int max_seq_len;
    int head_dim;
    float rope_theta;
    float norm_eps;
};

struct Parameters {
    Config config;
    std::vector<std::unordered_map<std::string, Tensor>> layers;
    Tokenizer tokenizer;

    static void* map_file(int fd);

    void load_config(nlohmann::json& header);
    void load_weights(void* p, nlohmann::json& header);
    void load_parameters(const std::string& path);

    // Overload [] operator so we can access layer Tensors directly, params[i]
    const std::unordered_map<std::string, Tensor>& operator[](size_t i) const {
        return layers[i];
    }
};

struct Model {
    std::shared_ptr<const Parameters> params;

    Model(std::shared_ptr<const Parameters> params) : params(params) {}
};