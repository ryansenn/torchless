#include "model.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

 Model::Model(std::string path){
    int fd = open(path.c_str(), 00);
    if (fd == -1) { std::cerr << ("open") << std::endl; return; }

    struct stat sb{};
    fstat(fd, &sb);

    // mapping the whole file in memory
    void* start = mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

    // first 8 bytes are size of json header
    uint64_t header_len = *static_cast<uint64_t*>(start);

    // load json payload
    std::string json_str(static_cast<char*>(start) + 8, header_len);
    nlohmann::json header = nlohmann::json::parse(json_str);

    // fill Config from "__metadata__"
     auto& m = header["__metadata__"];

     config.hidden_size       = std::stoi(m["hidden_size"].get<std::string>());
     config.intermediate_size = std::stoi(m["intermediate_size"].get<std::string>());
     config.n_layers          = std::stoi(m["n_layers"].get<std::string>());
     config.n_heads           = std::stoi(m["n_heads"].get<std::string>());
     config.n_kv_heads        = std::stoi(m["n_kv_heads"].get<std::string>());
     config.vocab_size        = std::stoi(m["vocab_size"].get<std::string>());
     config.max_seq_len       = std::stoi(m["max_seq_len"].get<std::string>());
     config.rope_theta        = std::stof(m["rope_theta"].get<std::string>());
     config.norm_eps          = std::stof(m["norm_eps"].get<std::string>());
}

