#include "model.h"
#include <iostream>

#include <fcntl.h>     // declares open()
#include <unistd.h>    // declares close()
#include <sys/mman.h>  // declares mmap()
#include <sys/stat.h>

// Memory-map the model file into virtual address space and return base pointer.
void* Parameters::map_file(int fd){
    // Get the size of the binary file
    struct stat st;
    if (fstat(fd, &st) < 0){
        std::cerr << "Model binary get size failed" << std::endl;
        std::exit(1);
    }

    size_t size = st.st_size;

    // Load the file into virtual memory
    void* p = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);

    if (p == MAP_FAILED){
        std::cerr << "Model mmap failed" << std::endl;
        std::exit(1);
    }

    return p;
}

// Extract model hyperparameters from JSON header into a Config struct.
void Parameters::load_config(nlohmann::json& header){
    nlohmann::json m = header["metadata"];

    config.hidden_size       = std::stoi(m["hidden_size"].get<std::string>());
    config.intermediate_size = std::stoi(m["intermediate_size"].get<std::string>());
    config.n_layers          = std::stoi(m["n_layers"].get<std::string>());
    config.n_heads           = std::stoi(m["n_heads"].get<std::string>());
    config.n_kv_heads        = std::stoi(m["n_kv_heads"].get<std::string>());
    config.vocab_size        = std::stoi(m["vocab_size"].get<std::string>());
    config.max_seq_len       = std::stoi(m["max_seq_len"].get<std::string>());
    config.rope_theta        = std::stof(m["rope_theta"].get<std::string>());
    config.norm_eps          = std::stof(m["norm_eps"].get<std::string>());
    config.head_dim          = config.hidden_size / config.n_heads;
}


// Load Tensor views for all weights using offsets from the JSON header.
void Parameters::load_weights(void* p, nlohmann::json& header){

}

void Parameters::load_parameters(const std::string& path){
    // Get file descriptor
    int fd = open(path.c_str(), O_RDONLY);

    if (fd < 0){
        std::cerr << "Model binary open failed" << std::endl;
        std::exit(1);
    }

    // Read the 8 byte header uint64_t size
    uint64_t header_size;
    read(fd, &header_size, sizeof(header_size));

    // Read and parse the JSON Header
    char header[header_size+1];
    read(fd, &header, header_size);
    header[header_size] = '\0';
    nlohmann::json header_json = nlohmann::json::parse(std::string(header));

    load_config(header_json);

    // Map file into virtual memory
    void* p = map_file(fd);

    // Load tensor weights to pointers in mmap
    //load_weights(p, header_json);

    close(fd);
}


