#include "model.h"
#include <fstream>

void Model::load(std::string path){
    std::ifstream f(path);
    f.seekg(0);
    
    char buffer[1024];
    uint8_t entry_type;
    std::string key;

    uint8_t value_type;


    while (f.peek() != EOF){
        f.read(buffer, 1);
        std::memcpy(&entry_type, buffer, sizeof(uint8_t));
        std::cout << entry_type << std::endl;

        // Metadata entry
        if (entry_type == 0) {
            f.read(buffer, 50);
            key.assign(buffer, 50);
            key.erase(key.find('\0'));

            f.read(buffer, 1);
            std::memcpy(&value_type, buffer, sizeof(uint8_t));

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
            f.read(buffer, 50);
            key.assign(buffer, 50);
            key.erase(key.find('\0'));

            

        }
        else {
            std::cerr << "FATAL: Unknown entry_type " << entry_type << " in model file\n";
            std::exit(EXIT_FAILURE);
        }
    }
}