#include "model.h"
#include <fstream>

void Model::load(std::string path){
    std::ifstream f(path);
    f.seekg(0);
    
    char buffer[1024];
    int32_t entry_type;
    int32_t key_len;
    std::string key;

    int32_t value_type;


    while (f.peek() != EOF){
        f.read(buffer, 4);
        std::memcpy(&entry_type, buffer, sizeof(int32_t));

        // Metadata entry
        if (entry_type == 0) {
            f.read(buffer, 4);
            std::memcpy(&key_len, buffer, sizeof(int32_t));

            f.read(buffer, key_len);
            key.assign(buffer, key_len);

            f.read(buffer, 4);
            std::memcpy(&value_type, buffer, sizeof(int32_t));

            switch(value_type){
                // int
                case 0:
                    int32_t value;
                    f.read(buffer, 4);
                    std::memcpy(&value, buffer, sizeof(int32_t));

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

        // Tensor entry
        else if (entry_type == 1) {

        }
    }
}