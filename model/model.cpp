#include "model.h"
#include <fstream>

void Model::load(std::string path){
    config = Config();

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Failed to open file" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    f.seekg(0);
    uint8_t entry_type;

    while (f.peek() != EOF){
        f.read(reinterpret_cast<char*>(&entry_type), sizeof(entry_type));

        // Metadata entry
        if (entry_type == 0) {
            load_metadata_entry(f);
        }

        // Tensor entry
        else if (entry_type == 1) {
            load_tensor_entry(f);
        }
        else {
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
    key.erase(key.find('\0'));

    f.read(reinterpret_cast<char*>(&value_type), sizeof(uint8_t));

    switch(value_type){
        // int
        case 0:
            int32_t value;
            f.read(reinterpret_cast<char*>(&value), sizeof(int32_t));

            if (key == "vocab_size") {
                config.vocab_size = value;
            } else if (key == "hidden_size") {
                config.hidden_size = value;
            }
            break;
        // float
        case 1:
            break;
        //string
        case 2:
            break;
    }
}

void Model::load_tensor_entry(std::ifstream& f){
    std::unique_ptr<Tensor> tensor;
    char buffer[1024];
    uint8_t value_type;
    std::string key;

    f.read(buffer, 50);
    key.assign(buffer, 50);
    key.erase(key.find('\0'));

    f.read(reinterpret_cast<char*>(&value_type), sizeof(value_type));

    int64_t size;
    f.read(reinterpret_cast<char*>(&size), sizeof(int64_t));

    void* data = std::malloc(size);
    f.read(reinterpret_cast<char*>(data), size);

    if (key == "vocab"){
        tokenizer = std::make_unique<Tokenizer>(reinterpret_cast<char*>(data), size);
    }

    tensor = std::make_unique<Tensor>(key, reinterpret_cast<float*>(data));
    tensor->shape[0] = size;

    if (key == "model.embed_tokens.weight"){
        token_embedding_table = std::move(tensor);
    }
}