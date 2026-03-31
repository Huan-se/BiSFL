#ifndef ENCLAVE_T_H__
#define ENCLAVE_T_H__

#include <stdint.h>
#include <wchar.h>
#include <stddef.h>
#include "sgx_edger8r.h" /* for sgx_ocall etc. */

#include "sgx_key_exchange.h"
#include "sgx_tcrypto.h"

#include <stdlib.h> /* for size_t */

#define SGX_CAST(type, item) ((type)(item))

#ifdef __cplusplus
extern "C" {
#endif

void ecall_set_verbose(int level);
void ecall_ra_keygen(uint8_t* out_pub_key, uint8_t* out_quote);
void ecall_ra_provision_seed(uint8_t* server_pub_key, uint8_t* cipher_payload);
void ecall_prepare_gradient(int client_id, const char* proj_seed_str, float* w_new, float* w_old, size_t model_len, int* ranges, size_t ranges_len, float* output_proj, size_t out_len);
void ecall_generate_masked_gradient_dynamic(const char* kappa_m_str, int t, const char* model_hash_str, int client_id, int* u1_ids, size_t u1_len, const char* k_weight_str, size_t model_len, int* ranges, size_t ranges_len, long long* output, size_t out_len);
void ecall_get_vector_shares_dynamic(const char* kappa_s_str, const char* kappa_m_str, int t, int* u1_ids, size_t u1_len, int* u2_ids, size_t u2_len, int my_id, long long* output_vector, size_t max_len);
void ecall_generate_noise_from_seed(const char* seed_str, size_t len, long long* output);

sgx_status_t SGX_CDECL ocall_print_string(const char* str);
sgx_status_t SGX_CDECL sgx_oc_cpuidex(int cpuinfo[4], int leaf, int subleaf);
sgx_status_t SGX_CDECL sgx_thread_wait_untrusted_event_ocall(int* retval, const void* self);
sgx_status_t SGX_CDECL sgx_thread_set_untrusted_event_ocall(int* retval, const void* waiter);
sgx_status_t SGX_CDECL sgx_thread_setwait_untrusted_events_ocall(int* retval, const void* waiter, const void* self);
sgx_status_t SGX_CDECL sgx_thread_set_multiple_untrusted_events_ocall(int* retval, const void** waiters, size_t total);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
