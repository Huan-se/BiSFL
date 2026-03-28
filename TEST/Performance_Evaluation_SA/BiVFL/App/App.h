#ifndef _APP_H_
#define _APP_H_
#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

int tee_init(const char* enclave_path);
void tee_destroy();

void tee_prepare_gradient(int client_id, int proj_seed, int param_size, const float* w_new, const float* w_old, float* output_proj);

void tee_generate_masked_gradient_sparse(
    const char* kappa_m_str, int t, const char* model_hash_str,
    int client_id, const float* w_new, float weight, int param_size, int64_t* out_masked_gradient
);

void tee_get_scalar_shares_sparse(
    const char* kappa_s_str, const char* kappa_m_str, int t, const char* view_hash_str,
    int client_id, const int* alive_neighbors, int num_alive,
    const int* dropped_neighbors, int num_dropped, int threshold,
    int64_t* out_shares, size_t max_len
);

#if defined(__cplusplus)
}
#endif
#endif /* !_APP_H_ */