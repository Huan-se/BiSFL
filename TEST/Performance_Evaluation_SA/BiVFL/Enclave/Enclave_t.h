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

void ecall_prepare_gradient(int client_id, int proj_seed, int param_size, const float* w_new, const float* w_old_cache, float* out_proj);
void ecall_generate_masked_gradient_sparse(const char* kappa_m_str, int t, const char* model_hash_str, int client_id, const float* w_new, float weight, int param_size, int64_t* out_masked_gradient);
void ecall_get_scalar_shares_sparse(const char* kappa_s_str, const char* kappa_m_str, int t, const char* view_hash_str, int client_id, const int* alive_neighbors, int num_alive, const int* dropped_neighbors, int num_dropped, int threshold, int64_t* out_shares, size_t max_len);

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
