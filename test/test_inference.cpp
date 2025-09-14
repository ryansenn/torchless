#include "context.h"

int test_kv_cache(){
    InferenceState inferenceState(get_model());


    return 0;
}

RegisterTest reg_test_kv_cache("test kv cache", &test_kv_cache);