#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sgx_urts.h"
#include "App.h"
#include "Enclave_u.h"

sgx_enclave_id_t global_eid = 0;

void print_error_message(sgx_status_t ret) { printf("[Bridge Error] SGX Status: 0x%X\n", ret); }
void ocall_print_string(const char *str) { printf("%s", str); }

extern "C" {

int tee_init(const char* enclave_path) {
    if (global_eid != 0) return 0;
    sgx_status_t ret = sgx_create_enclave(enclave_path, SGX_DEBUG_FLAG, NULL, NULL, &global_eid, NULL);
    if (ret != SGX_SUCCESS) { print_error_message(ret); return -1; }
    return 0;
}

void tee_destroy() {
    if (global_eid != 0) { sgx_destroy_enclave(global_eid); global_eid = 0; }
}

void tee_prepare_gradient(int client_id, int proj_seed, int param_size, const float* w_new, const float* w_old, float* output_proj) {
    ecall_prepare_gradient(global_eid, client_id, proj_seed, param_size, w_new, w_old, output_proj);
}

void tee_generate_masked_gradient_sparse(
    const char* kappa_m_str, int t, const char* model_hash_str,
    int client_id, const float* w_new, float weight, int param_size, int64_t* out_masked_gradient
) {
    ecall_generate_masked_gradient_sparse(global_eid, kappa_m_str, t, model_hash_str, client_id, w_new, weight, param_size, out_masked_gradient);
}

void tee_get_scalar_shares_sparse(
    const char* kappa_s_str, const char* kappa_m_str, int t, const char* view_hash_str,
    int client_id, const int* alive_neighbors, int num_alive,
    const int* dropped_neighbors, int num_dropped, int threshold,
    int64_t* out_shares, size_t max_len
) {
    ecall_get_scalar_shares_sparse(global_eid, kappa_s_str, kappa_m_str, t, view_hash_str, client_id, alive_neighbors, num_alive, dropped_neighbors, num_dropped, threshold, out_shares, max_len);
}

} // extern "C"