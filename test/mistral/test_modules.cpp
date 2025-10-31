#include "setup/context.h"


int test_embedding() {
    // table: 4 embeddings, dim=3
    Tensor table({
                         0.1f, 0.2f, 0.3f,
                         0.4f, 0.5f, 0.6f,
                         0.7f, 0.8f, 0.9f,
                         1.0f, 1.1f, 1.2f
                 }, {4, 3});

    Embedding emb(table);

    std::vector<size_t> idx{3, 1, 3, 0};
    Tensor out = emb.forward(idx);

    Tensor expected({
                            1.0f, 1.1f, 1.2f,
                            0.4f, 0.5f, 0.6f,
                            1.0f, 1.1f, 1.2f,
                            0.1f, 0.2f, 0.3f
                    }, {4, 3});

    if (!equals(out, expected)) {
        std::cout << "Embedding mismatch" << std::endl;
        return 1;
    }
    return 0;
}

RegisterTest embedding_reg("test embedding", &test_embedding);

int test_rmsnorm() {
    Tensor x({2.4f, 7.8f, 1.1f, 5.3f, 9.0f, 4.7f, 6.2f, 3.8f, 8.5f, 0.6f}, {10});
    Tensor g({0.8f, 1.2f, 1.0f, 0.9f, 1.1f, 1.3f, 0.7f, 1.0f, 1.2f, 0.95f}, {10});

    RMSNorm rms(g);
    Tensor out = rms.forward(x);

    Tensor expected({0.3371f, 1.6432f, 0.1931f, 0.8374f, 1.7380f, 1.0726f, 0.7619f, 0.6671f, 1.7906f, 0.1001f}, {10});

    if (!equals(out, expected)) {
        std::cout << "RMSNorm mismatch" << std::endl;
        return 1;
    }

    return 0;
}

RegisterTest rmsnorm_reg("test rmsnorm", &test_rmsnorm);


int test_rotary_embedding() {
    static const float COS_POS0[64] = {
            1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,
            1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,
            1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,
            1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f
    };

    static const float COS_POS3[64] = {
            -0.989992f,-0.855801f,-0.627927f,-0.368457f,-0.115966f,0.109673f,0.300967f,0.457582f,
            0.582800f,0.681190f,0.752900f,0.809620f,0.861000f,0.895200f,0.921000f,0.940600f,
            0.955300f,0.966400f,0.974800f,0.981100f,0.985800f,0.989300f,0.992000f,0.994000f,
            0.995500f,0.996600f,0.997500f,0.998100f,0.998600f,0.998900f,0.999200f,0.999400f,
            0.999600f,0.999700f,0.999700f,0.999800f,0.999900f,0.999900f,0.999900f,1.000000f,
            1.000000f,1.000000f,1.000000f,1.000000f,1.000000f,1.000000f,1.000000f,1.000000f,
            1.000000f,1.000000f,1.000000f,1.000000f,1.000000f,1.000000f,1.000000f,1.000000f,
            1.000000f,1.000000f,1.000000f,1.000000f,1.000000f,1.000000f,1.000000f,1.000000f
    };

    static const float SIN_POS0[64] = {
            0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
            0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
            0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
            0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f
    };

    static const float SIN_POS3[64] = {
            0.141120f,0.517306f,0.778273f,0.929645f,0.993253f,0.993970f,0.953630f,0.889170f,
            0.812650f,0.732190f,0.652900f,0.577820f,0.508540f,0.445720f,0.389470f,0.339550f,
            0.295520f,0.256880f,0.223080f,0.193580f,0.167900f,0.145570f,0.126170f,0.109330f,
            0.094726f,0.082060f,0.071081f,0.061567f,0.053323f,0.046181f,0.039995f,0.034637f,
            0.029995f,0.025976f,0.022495f,0.019480f,0.016869f,0.014609f,0.012651f,0.010955f,
            0.009487f,0.008215f,0.007114f,0.006161f,0.005335f,0.004620f,0.004001f,0.003464f,
            0.003000f,0.002598f,0.002250f,0.001948f,0.001687f,0.001461f,0.001265f,0.001096f,
            0.000949f,0.000822f,0.000711f,0.000616f,0.000533f,0.000462f,0.000400f,0.000346f
    };

    Tensor exp_cos0(const_cast<float *>(COS_POS0), {64});
    Tensor exp_cos3(const_cast<float *>(COS_POS3), {64});
    Tensor exp_sin0(const_cast<float *>(SIN_POS0), {64});
    Tensor exp_sin3(const_cast<float *>(SIN_POS3), {64});

    Model m = get_model();
    RotaryEmbedding r(m.params->config);
    Tensor emb = r.forward({0,3});

    Tensor cos = emb.at({0});
    Tensor sin = emb.at({1});

    Tensor got_cos0 = cos.at({0});
    Tensor got_cos3 = cos.at({1});
    Tensor got_sin0 = sin.at({0});
    Tensor got_sin3 = sin.at({1});

    if (!equals(got_cos0, exp_cos0) ||
        !equals(got_cos3, exp_cos3) ||
        !equals(got_sin0, exp_sin0) ||
        !equals(got_sin3, exp_sin3)) {
        std::cout << "Rotary embedding mismatch" << std::endl;
        return 1;
    }

    return 0;
}

RegisterTest rotary_reg("test rotary embedding", &test_rotary_embedding);
