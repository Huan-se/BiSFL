/* App/App.cpp */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sgx_urts.h"
#include "App.h"
#include "Enclave_u.h"

sgx_enclave_id_t global_eid = 0;

void print_error_message(sgx_status_t ret) {
    printf("[Bridge Error] SGX Status: 0x%X\n", ret);
}

void ocall_print_string(const char *str) {
    printf("%s", str);
}

extern "C" {

void tee_set_verbose(int level) {
    if (global_eid == 0) return; 
    ecall_set_verbose(global_eid, level);
}

int tee_init(const char* enclave_path) {
    if (global_eid != 0) return 0;
    sgx_status_t ret = sgx_create_enclave(enclave_path, SGX_DEBUG_FLAG, NULL, NULL, &global_eid, NULL);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        return -1;
    }
    return 0;
}

void tee_destroy() {
    if (global_eid != 0) {
        sgx_destroy_enclave(global_eid);
        global_eid = 0;
    }
}

void tee_prepare_gradient(
    int client_id, 
    const char* proj_seed_str, 
    float* w_new, float* w_old, size_t model_len, 
    int* ranges, size_t ranges_len, float* output_proj, size_t out_len
) {
    sgx_status_t ret = ecall_prepare_gradient(
        global_eid, client_id, proj_seed_str, 
        w_new, w_old, model_len, ranges, ranges_len, output_proj, out_len
    );
    if (ret != SGX_SUCCESS) print_error_message(ret);
}

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
) {
    sgx_status_t ret = ecall_generate_masked_gradient_dynamic(
        global_eid, kappa_m_str, t, model_hash_str, client_id, 
        u1_ids, u1_len, k_weight_str, 
        model_len, ranges, ranges_len, output, out_len
    );
    if (ret != SGX_SUCCESS) print_error_message(ret);
}

void tee_get_vector_shares_dynamic(
    const char* kappa_s_str, 
    const char* kappa_m_str, 
    int t,
    int* u1_ids, size_t u1_len, 
    int* u2_ids, size_t u2_len, 
    int my_id, 
    long long* output_vector, 
    size_t max_len
) {
    sgx_status_t ret = ecall_get_vector_shares_dynamic(
        global_eid, kappa_s_str, kappa_m_str, t,
        u1_ids, u1_len, u2_ids, u2_len, my_id, 
        output_vector, max_len
    );
    if (ret != SGX_SUCCESS) print_error_message(ret);
}

void tee_generate_noise_from_seed(
    const char* seed_str, 
    size_t len, 
    long long* output
) {
    sgx_status_t ret = ecall_generate_noise_from_seed(
        global_eid, seed_str, len, output
    );
    if (ret != SGX_SUCCESS) print_error_message(ret);
}

} // extern "C"