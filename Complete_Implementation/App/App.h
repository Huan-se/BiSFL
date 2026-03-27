/* App/App.h */
#ifndef _APP_H_
#define _APP_H_

#include <stddef.h> /* for size_t */

#if defined(__cplusplus)
extern "C" {
#endif

int tee_init(const char* enclave_path);
void tee_destroy();

// Phase 2
void tee_prepare_gradient(
    int client_id, 
    const char* proj_seed_str,  
    float* w_new, float* w_old, size_t model_len, 
    int* ranges, size_t ranges_len, float* output_proj, size_t out_len
);

// Phase 4
void tee_generate_masked_gradient_dynamic(
    const char* kappa_m_str,
    int t,
    const char* model_hash_str,
    int client_id, 
    int* u1_ids, size_t u1_len,
    const char* k_weight_str,       
    size_t model_len, 
    int* ranges, size_t ranges_len, 
    long long* output, size_t out_len
);

// Phase 5
void tee_get_vector_shares_dynamic(
    const char* kappa_s_str,       
    const char* kappa_m_str, 
    int t,
    int* u1_ids, size_t u1_len, 
    int* u2_ids, size_t u2_len, 
    int my_id, 
    long long* output_vector, 
    size_t max_len
);

// Noise
void tee_generate_noise_from_seed(
    const char* seed_str,           
    size_t len, 
    long long* output
);

#if defined(__cplusplus)
}
#endif

#endif /* !_APP_H_ */