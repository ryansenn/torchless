#include "context.h"

int test_kv_cache() {
    InferenceState st(get_model());
    const int H = st.model.config.n_kv_heads;
    const int D = st.model.config.head_dim;

    // hijack the hidden state with ones
    std::vector<float> ones(st.model.config.hidden_size, 1.f);
    st.x.copy_from(ones.data(), ones.size() * sizeof(float));

    // push all layers at pos 0
    for (int i = 0; i < st.model.config.n_layers; ++i) st.push_kv(i);
    st.pos++;

    if (st.pos != 1) {
        std::cout << "pos mismatch after push: got " << st.pos << ", want 1\n";
        return 1;
    }

    // check layer 0 @ pos 0
    st.push_kv(0);
    {
        float a = st.k_cache[0]->at({0, 0}).data[0];       // head 0, dim 0
        float b = st.k_cache[0]->at({H-1, 0}).data[D-1];   // last head, last dim
        if (!equals(a, 0.00145736f)) {
            std::cout << "k_cache[0][h=0,pos=0,d=0] mismatch: got " << a << ", want 0.00145736\n";
            return 1;
        }
        if (!equals(b, -0.04015780f)) {
            std::cout << "k_cache[0][h=" << (H-1) << ",pos=0,d=" << (D-1)
                      << "] mismatch: got " << b << ", want -0.04015780\n";
            return 1;
        }
    }

    // check layer 31 @ pos 0
    st.push_kv(31);
    {
        float a = st.k_cache[31]->at({0, 0}).data[0];
        float b = st.k_cache[31]->at({H-1, 0}).data[D-1];
        if (!equals(a, 0.11939027f)) {
            std::cout << "k_cache[31][h=0,pos=0,d=0] mismatch: got " << a << ", want 0.11939027\n";
            return 1;
        }
        if (!equals(b, -0.11372215f)) {
            std::cout << "k_cache[31][h=" << (H-1) << ",pos=0,d=" << (D-1)
                      << "] mismatch: got " << b << ", want -0.11372215\n";
            return 1;
        }
    }

    // push again → pos 1
    for (int i = 0; i < st.model.config.n_layers; ++i) st.push_kv(i);
    st.pos++;

    // check layer 0 @ pos 1 (same expected refs as before)
    {
        float a = st.k_cache[0]->at({0, 1}).data[0];
        float b = st.k_cache[0]->at({H-1, 1}).data[D-1];
        if (!equals(a, 0.00145736f)) {
            std::cout << "k_cache[0][h=0,pos=1,d=0] mismatch: got " << a << ", want 0.00145736\n";
            return 1;
        }
        if (!equals(b, -0.04015780f)) {
            std::cout << "k_cache[0][h=" << (H-1) << ",pos=1,d=" << (D-1)
                      << "] mismatch: got " << b << ", want -0.04015780\n";
            return 1;
        }
    }

    return 0;
}

RegisterTest reg_test_kv_cache("test kv cache", &test_kv_cache);