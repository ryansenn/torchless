#include <unordered_map>
#include <string>
#include "tensor.h"


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
    // Pointer to start of model binary file
    void* start;

    const std::vector<std::unordered_map<std::string, Tensor>> layers;
    const Config config;

    Parameters(const std::string& path) : start(map_file(path)), layers(load_weights_from_path(path)), config(load_config_from_path(path)) {}

    static void* map_file(const std::string& path);
    static std::vector<std::unordered_map<std::string, Tensor>> load_weights_from_path(const std::string& path);
    static Config load_config_from_path(const std::string& path);

    // Overload [] operator so we can access layer Tensors directly, params[i]
    const std::unordered_map<std::string, Tensor>& operator[](size_t i) const {
        return layers[i];
    }
};

struct Model {
    std::shared_ptr<const Parameters> params;

    Model(const std::string& path) :
};