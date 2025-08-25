#include "model.h"
#include <fstream>
#include <unordered_map>
#include <functional>

void Model::load(std::string path){
    config = Config();

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Failed to open file" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    uint8_t entry_type;
    while (f.peek() != EOF){
        f.read(reinterpret_cast<char*>(&entry_type), sizeof(entry_type));
        if (!f) break;

        if (entry_type == 0) {
            load_metadata_entry(f);
        } else if (entry_type == 1) {
            load_tensor_entry(f);
        } else {
            std::cerr << "FATAL: Unknown entry_type " << entry_type << " in model file\n";
            std::exit(EXIT_FAILURE);
        }
    }
}

void Model::load_metadata_entry(std::ifstream& f){
    char buffer[1024];
    uint8_t value_type;
    std::string key;

    f.read(buffer, 50);
    key.assign(buffer, 50);
    size_t pos = key.find('\0');
    if (pos != std::string::npos) key.erase(pos);

    f.read(reinterpret_cast<char*>(&value_type), sizeof(uint8_t));

    static const std::unordered_map<std::string, std::function<void(Config&, int32_t)>> int_setters = {
            {"vocab_size",        [](Config& c, int32_t v){ c.vocab_size = v; }},
            {"hidden_size",       [](Config& c, int32_t v){ c.hidden_size = v; }},
            {"num_hidden_layers", [](Config& c, int32_t v){ c.num_hidden_layers = v; }}
    };

    switch(value_type){
        case 0: {
            int32_t value;
            f.read(reinterpret_cast<char*>(&value), sizeof(value));
            auto it = int_setters.find(key);
            if (it != int_setters.end()) it->second(config, value);
            break;
        }
        case 1: {
            float value;
            f.read(reinterpret_cast<char*>(&value), sizeof(value));
            break;
        }
        case 2: {
            int32_t len;
            f.read(reinterpret_cast<char*>(&len), sizeof(len));
            f.ignore(len);
            break;
        }
        default:
            std::cerr << "Unknown metadata value_type " << (int)value_type << "\n";
            break;
    }
}

void Model::load_tensor_entry(std::ifstream& f){
    char buffer[1024];
    uint8_t value_type;
    std::string key;

    f.read(buffer, 50);
    key.assign(buffer, 50);
    size_t pos = key.find('\0');
    if (pos != std::string::npos) key.erase(pos);

    f.read(reinterpret_cast<char*>(&value_type), sizeof(value_type));

    int64_t size;
    f.read(reinterpret_cast<char*>(&size), sizeof(int64_t));

    void* data = std::malloc(size);
    f.read(reinterpret_cast<char*>(data), size);

    if (key == "vocab"){
        tokenizer = std::make_unique<Tokenizer>(reinterpret_cast<char*>(data), size);
        return;
    }

    auto tensor = std::make_unique<Tensor>(key, reinterpret_cast<float*>(data));
    tensor->shape[0] = size / sizeof(float);

    static const std::unordered_map<std::string, std::function<void(Model&, std::unique_ptr<Tensor>)>> tensor_setters = {
            {"model.embed_tokens.weight", [](Model& m, std::unique_ptr<Tensor> t){ m.token_embedding_table = std::move(t); }}
    };

    auto it = tensor_setters.find(key);
    if (it != tensor_setters.end()) {
        it->second(*this, std::move(tensor));
    }
}