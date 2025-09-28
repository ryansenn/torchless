#include "context.h"

int test_kv_cache() {
    InferenceState st(get_model());
    const int H = st.model.config.n_kv_heads;
    const int D = st.model.config.head_dim;

    std::vector<float> ones(st.model.config.hidden_size, 1.f);
    st.x.copy_from(ones.data(), ones.size() * sizeof(float));

    for (int i = 0; i < st.model.config.n_layers; ++i) st.push_kv(i);
    st.pos++;

    if (st.pos != 1) {
        std::cout << "pos mismatch after push: got " << st.pos << ", want 1\n";
        return 1;
    }

    st.push_kv(0);
    {
        float a = st.k_cache[0]->at({0, st.pos}).data[0];
        float b = st.k_cache[0]->at({H-1, st.pos}).data[D-1];
        float a_ref = st.k.data[0];
        float b_ref = st.k.data[H * D - 1];
        if (!equals(a, a_ref)) {
            std::cout << "k_cache[0][h=0,pos=" << st.pos << ",d=0] mismatch: got " << a << ", want " << a_ref << "\n";
            return 1;
        }
        if (!equals(b, b_ref)) {
            std::cout << "k_cache[0][h=" << (H-1) << ",pos=" << st.pos << ",d=" << (D-1)
                      << "] mismatch: got " << b << ", want " << b_ref << "\n";
            return 1;
        }
    }

    st.push_kv(31);
    {
        float a = st.k_cache[31]->at({0, st.pos}).data[0];
        float b = st.k_cache[31]->at({H-1, st.pos}).data[D-1];
        float a_ref = st.k.data[0];
        float b_ref = st.k.data[H * D - 1];
        if (!equals(a, a_ref)) {
            std::cout << "k_cache[31][h=0,pos=" << st.pos << ",d=0] mismatch: got " << a << ", want " << a_ref << "\n";
            return 1;
        }
        if (!equals(b, b_ref)) {
            std::cout << "k_cache[31][h=" << (H-1) << ",pos=" << st.pos << ",d=" << (D-1)
                      << "] mismatch: got " << b << ", want " << b_ref << "\n";
            return 1;
        }
    }

    for (int i = 0; i < st.model.config.n_layers; ++i) st.push_kv(i);
    st.pos++;

    st.push_kv(0);
    {
        float a = st.k_cache[0]->at({0, st.pos}).data[0];
        float b = st.k_cache[0]->at({H-1, st.pos}).data[D-1];
        float a_ref = st.k.data[0];
        float b_ref = st.k.data[H * D - 1];
        if (!equals(a, a_ref)) {
            std::cout << "k_cache[0][h=0,pos=" << st.pos << ",d=0] mismatch: got " << a << ", want " << a_ref << "\n";
            return 1;
        }
        if (!equals(b, b_ref)) {
            std::cout << "k_cache[0][h=" << (H-1) << ",pos=" << st.pos << ",d=" << (D-1)
                      << "] mismatch: got " << b << ", want " << b_ref << "\n";
            return 1;
        }
    }

    return 0;
}

RegisterTest reg_test_kv_cache("test kv cache", &test_kv_cache);