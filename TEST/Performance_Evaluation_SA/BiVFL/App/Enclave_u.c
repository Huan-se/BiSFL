#include "Enclave_u.h"
#include <errno.h>

typedef struct ms_ecall_prepare_gradient_t {
	int ms_client_id;
	int ms_proj_seed;
	int ms_param_size;
	const float* ms_w_new;
	const float* ms_w_old_cache;
	float* ms_out_proj;
} ms_ecall_prepare_gradient_t;

typedef struct ms_ecall_generate_masked_gradient_sparse_t {
	const char* ms_kappa_m_str;
	size_t ms_kappa_m_str_len;
	int ms_t;
	const char* ms_model_hash_str;
	size_t ms_model_hash_str_len;
	int ms_client_id;
	const float* ms_w_new;
	float ms_weight;
	int ms_param_size;
	int64_t* ms_out_masked_gradient;
} ms_ecall_generate_masked_gradient_sparse_t;

typedef struct ms_ecall_get_scalar_shares_sparse_t {
	const char* ms_kappa_s_str;
	size_t ms_kappa_s_str_len;
	const char* ms_kappa_m_str;
	size_t ms_kappa_m_str_len;
	int ms_t;
	const char* ms_view_hash_str;
	size_t ms_view_hash_str_len;
	int ms_client_id;
	const int* ms_alive_neighbors;
	int ms_num_alive;
	const int* ms_dropped_neighbors;
	int ms_num_dropped;
	int ms_threshold;
	int64_t* ms_out_shares;
	size_t ms_max_len;
} ms_ecall_get_scalar_shares_sparse_t;

typedef struct ms_ocall_print_string_t {
	const char* ms_str;
} ms_ocall_print_string_t;

typedef struct ms_sgx_oc_cpuidex_t {
	int* ms_cpuinfo;
	int ms_leaf;
	int ms_subleaf;
} ms_sgx_oc_cpuidex_t;

typedef struct ms_sgx_thread_wait_untrusted_event_ocall_t {
	int ms_retval;
	const void* ms_self;
} ms_sgx_thread_wait_untrusted_event_ocall_t;

typedef struct ms_sgx_thread_set_untrusted_event_ocall_t {
	int ms_retval;
	const void* ms_waiter;
} ms_sgx_thread_set_untrusted_event_ocall_t;

typedef struct ms_sgx_thread_setwait_untrusted_events_ocall_t {
	int ms_retval;
	const void* ms_waiter;
	const void* ms_self;
} ms_sgx_thread_setwait_untrusted_events_ocall_t;

typedef struct ms_sgx_thread_set_multiple_untrusted_events_ocall_t {
	int ms_retval;
	const void** ms_waiters;
	size_t ms_total;
} ms_sgx_thread_set_multiple_untrusted_events_ocall_t;

static sgx_status_t SGX_CDECL Enclave_ocall_print_string(void* pms)
{
	ms_ocall_print_string_t* ms = SGX_CAST(ms_ocall_print_string_t*, pms);
	ocall_print_string(ms->ms_str);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_sgx_oc_cpuidex(void* pms)
{
	ms_sgx_oc_cpuidex_t* ms = SGX_CAST(ms_sgx_oc_cpuidex_t*, pms);
	sgx_oc_cpuidex(ms->ms_cpuinfo, ms->ms_leaf, ms->ms_subleaf);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_sgx_thread_wait_untrusted_event_ocall(void* pms)
{
	ms_sgx_thread_wait_untrusted_event_ocall_t* ms = SGX_CAST(ms_sgx_thread_wait_untrusted_event_ocall_t*, pms);
	ms->ms_retval = sgx_thread_wait_untrusted_event_ocall(ms->ms_self);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_sgx_thread_set_untrusted_event_ocall(void* pms)
{
	ms_sgx_thread_set_untrusted_event_ocall_t* ms = SGX_CAST(ms_sgx_thread_set_untrusted_event_ocall_t*, pms);
	ms->ms_retval = sgx_thread_set_untrusted_event_ocall(ms->ms_waiter);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_sgx_thread_setwait_untrusted_events_ocall(void* pms)
{
	ms_sgx_thread_setwait_untrusted_events_ocall_t* ms = SGX_CAST(ms_sgx_thread_setwait_untrusted_events_ocall_t*, pms);
	ms->ms_retval = sgx_thread_setwait_untrusted_events_ocall(ms->ms_waiter, ms->ms_self);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_sgx_thread_set_multiple_untrusted_events_ocall(void* pms)
{
	ms_sgx_thread_set_multiple_untrusted_events_ocall_t* ms = SGX_CAST(ms_sgx_thread_set_multiple_untrusted_events_ocall_t*, pms);
	ms->ms_retval = sgx_thread_set_multiple_untrusted_events_ocall(ms->ms_waiters, ms->ms_total);

	return SGX_SUCCESS;
}

static const struct {
	size_t nr_ocall;
	void * table[6];
} ocall_table_Enclave = {
	6,
	{
		(void*)Enclave_ocall_print_string,
		(void*)Enclave_sgx_oc_cpuidex,
		(void*)Enclave_sgx_thread_wait_untrusted_event_ocall,
		(void*)Enclave_sgx_thread_set_untrusted_event_ocall,
		(void*)Enclave_sgx_thread_setwait_untrusted_events_ocall,
		(void*)Enclave_sgx_thread_set_multiple_untrusted_events_ocall,
	}
};
sgx_status_t ecall_prepare_gradient(sgx_enclave_id_t eid, int client_id, int proj_seed, int param_size, const float* w_new, const float* w_old_cache, float* out_proj)
{
	sgx_status_t status;
	ms_ecall_prepare_gradient_t ms;
	ms.ms_client_id = client_id;
	ms.ms_proj_seed = proj_seed;
	ms.ms_param_size = param_size;
	ms.ms_w_new = w_new;
	ms.ms_w_old_cache = w_old_cache;
	ms.ms_out_proj = out_proj;
	status = sgx_ecall(eid, 0, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_generate_masked_gradient_sparse(sgx_enclave_id_t eid, const char* kappa_m_str, int t, const char* model_hash_str, int client_id, const float* w_new, float weight, int param_size, int64_t* out_masked_gradient)
{
	sgx_status_t status;
	ms_ecall_generate_masked_gradient_sparse_t ms;
	ms.ms_kappa_m_str = kappa_m_str;
	ms.ms_kappa_m_str_len = kappa_m_str ? strlen(kappa_m_str) + 1 : 0;
	ms.ms_t = t;
	ms.ms_model_hash_str = model_hash_str;
	ms.ms_model_hash_str_len = model_hash_str ? strlen(model_hash_str) + 1 : 0;
	ms.ms_client_id = client_id;
	ms.ms_w_new = w_new;
	ms.ms_weight = weight;
	ms.ms_param_size = param_size;
	ms.ms_out_masked_gradient = out_masked_gradient;
	status = sgx_ecall(eid, 1, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_get_scalar_shares_sparse(sgx_enclave_id_t eid, const char* kappa_s_str, const char* kappa_m_str, int t, const char* view_hash_str, int client_id, const int* alive_neighbors, int num_alive, const int* dropped_neighbors, int num_dropped, int threshold, int64_t* out_shares, size_t max_len)
{
	sgx_status_t status;
	ms_ecall_get_scalar_shares_sparse_t ms;
	ms.ms_kappa_s_str = kappa_s_str;
	ms.ms_kappa_s_str_len = kappa_s_str ? strlen(kappa_s_str) + 1 : 0;
	ms.ms_kappa_m_str = kappa_m_str;
	ms.ms_kappa_m_str_len = kappa_m_str ? strlen(kappa_m_str) + 1 : 0;
	ms.ms_t = t;
	ms.ms_view_hash_str = view_hash_str;
	ms.ms_view_hash_str_len = view_hash_str ? strlen(view_hash_str) + 1 : 0;
	ms.ms_client_id = client_id;
	ms.ms_alive_neighbors = alive_neighbors;
	ms.ms_num_alive = num_alive;
	ms.ms_dropped_neighbors = dropped_neighbors;
	ms.ms_num_dropped = num_dropped;
	ms.ms_threshold = threshold;
	ms.ms_out_shares = out_shares;
	ms.ms_max_len = max_len;
	status = sgx_ecall(eid, 2, &ocall_table_Enclave, &ms);
	return status;
}

