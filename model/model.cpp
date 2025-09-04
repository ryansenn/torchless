#include "model.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

 std::unique_ptr<Tensor> Model::load_tensor_by_key(const nlohmann::json& header, const std::string& key) {
    uint64_t start = header[key]["data_offsets"][0].get<uint64_t>();
    uint64_t end = header[key]["data_offsets"][1].get<uint64_t>();
    std::vector<int64_t> shape = header[key]["shape"].get<std::vector<int64_t>>();

    return std::make_unique<Tensor>(key, reinterpret_cast<float *>(base_offset + start));
}

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

    base_offset = static_cast<uint8_t*>(start) + 8 + header_len;

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

     // Load pointers to tensors
     token_embedding_table = load_tensor_by_key(header, "model.embed_tokens.weight");

     // Load vocab
     int64_t begin = header["vocab"]["data_offsets"][0].get<uint64_t>();
     uint64_t end = header["vocab"]["data_offsets"][1].get<uint64_t>();
     char* raw_vocab = reinterpret_cast<char *>(base_offset + begin);
     tokenizer = std::make_unique<Tokenizer>(raw_vocab, end-begin);

     /*
     // attention proj weights
     for (int i=0; i<config.n_layers; i++){
         Block block;
         const std::string base = "model.layers." + std::to_string(i) + ".self_attn.";
         block.wq = load_tensor_by_key(header, base + "q_proj.weight");
         block.wk = load_tensor_by_key(header, base + "k_proj.weight");
         block.wv = load_tensor_by_key(header, base + "v_proj.weight");
         block.wo = load_tensor_by_key(header, base + "o_proj.weight");
         blocks.push_back(std::move(block));
     }
     */
}

