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
    if (header.contains("__metadata__")) {
        const auto& m = header["__metadata__"];
        auto get_i = [&](const char* k){ return m.contains(k) ? std::stoi(m[k].get<std::string>()) : 0; };
        config.hidden_size       = get_i("hidden_dim");
        config.num_hidden_layers = get_i("n_layers");
        config.vocab_size        = get_i("vocab_size");
    }
}

