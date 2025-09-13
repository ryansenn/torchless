struct InferenceState {
    Tensor x; // [hidden_size]

    Tensor q;
    Tensor k;
    Tensor v;

    // KV cache
    // Each block stores Tensor of size [max_seq_len, n_kv_heads, head_dim]
    std::vector<std::unique_ptr<Tensor>> k_cache;
    std::vector<std::unique_ptr<Tensor>> v_cache;

    int pos;

    InferenceState(Config& config) :
            x("x", {config.hidden_size}),
            q("q", {config.hidden_size}),
            k("k", {config.n_kv_heads * config.hidden_size / config.n_heads}),
            v("v", {config.n_kv_heads * config.hidden_size / config.n_heads}),
            pos(0) {

        // Initialize empty KV cache
        int64_t head_dim = config.hidden_size / config.n_heads;
        std::vector<int64_t> shape = {config.max_seq_len, config.n_kv_heads, head_dim};

        for (int i=0; i<config.n_layers; i++){
            std::unique_ptr<Tensor> k = std::make_unique<Tensor>("k_cache_" + std::to_string(i), shape);
            std::unique_ptr<Tensor> v = std::make_unique<Tensor>("v_cache_" + std::to_string(i), shape);

            k_cache.push_back(std::move(k));
            v_cache.push_back(std::move(v));
        }
    }
};