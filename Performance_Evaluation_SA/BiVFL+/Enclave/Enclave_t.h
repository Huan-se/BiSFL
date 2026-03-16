#ifndef ENCLAVE_T_H__
#define ENCLAVE_T_H__

#include <stdint.h>
#include <wchar.h>
#include <stddef.h>
#include "sgx_edger8r.h" /* for sgx_ocall etc. */


#include <stdlib.h> /* for size_t */

#define SGX_CAST(type, item) ((type)(item))

#ifdef __cplusplus
extern "C" {
#endif

void ecall_generate_masked_gradient_sparse(uint32_t root_seed, uint32_t global_seed, int client_id, const int* active_neighbors, int num_neighbors, float weight, int param_size, int64_t* out_masked_gradient);
void ecall_get_scalar_shares_sparse(uint32_t seed_sss, uint32_t root_seed, const int* dropped_neighbors, int num_dropped, int client_id, int threshold, uint64_t* out_shares);
void ecall_prepare_gradient(int client_id, int proj_seed, int param_size, const float* w_new, const float* w_old_cache, float* out_proj);


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
