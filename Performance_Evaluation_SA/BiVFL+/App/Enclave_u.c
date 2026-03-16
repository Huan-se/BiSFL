#include "Enclave_u.h"
#include <errno.h>

typedef struct ms_ecall_generate_masked_gradient_sparse_t {
	uint32_t ms_root_seed;
	uint32_t ms_global_seed;
	int ms_client_id;
	const int* ms_active_neighbors;
	int ms_num_neighbors;
	float ms_weight;
	int ms_param_size;
	int64_t* ms_out_masked_gradient;
} ms_ecall_generate_masked_gradient_sparse_t;

typedef struct ms_ecall_get_scalar_shares_sparse_t {
	uint32_t ms_seed_sss;
	uint32_t ms_root_seed;
	const int* ms_dropped_neighbors;
	int ms_num_dropped;
	int ms_client_id;
	int ms_threshold;
	uint64_t* ms_out_shares;
} ms_ecall_get_scalar_shares_sparse_t;

typedef struct ms_ecall_prepare_gradient_t {
	int ms_client_id;
	int ms_proj_seed;
	int ms_param_size;
	const float* ms_w_new;
	const float* ms_w_old_cache;
	float* ms_out_proj;
} ms_ecall_prepare_gradient_t;

static const struct {
	size_t nr_ocall;
	void * table[1];
} ocall_table_Enclave = {
	0,
	{ NULL },
};
sgx_status_t ecall_generate_masked_gradient_sparse(sgx_enclave_id_t eid, uint32_t root_seed, uint32_t global_seed, int client_id, const int* active_neighbors, int num_neighbors, float weight, int param_size, int64_t* out_masked_gradient)
{
	sgx_status_t status;
	ms_ecall_generate_masked_gradient_sparse_t ms;
	ms.ms_root_seed = root_seed;
	ms.ms_global_seed = global_seed;
	ms.ms_client_id = client_id;
	ms.ms_active_neighbors = active_neighbors;
	ms.ms_num_neighbors = num_neighbors;
	ms.ms_weight = weight;
	ms.ms_param_size = param_size;
	ms.ms_out_masked_gradient = out_masked_gradient;
	status = sgx_ecall(eid, 0, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_get_scalar_shares_sparse(sgx_enclave_id_t eid, uint32_t seed_sss, uint32_t root_seed, const int* dropped_neighbors, int num_dropped, int client_id, int threshold, uint64_t* out_shares)
{
	sgx_status_t status;
	ms_ecall_get_scalar_shares_sparse_t ms;
	ms.ms_seed_sss = seed_sss;
	ms.ms_root_seed = root_seed;
	ms.ms_dropped_neighbors = dropped_neighbors;
	ms.ms_num_dropped = num_dropped;
	ms.ms_client_id = client_id;
	ms.ms_threshold = threshold;
	ms.ms_out_shares = out_shares;
	status = sgx_ecall(eid, 1, &ocall_table_Enclave, &ms);
	return status;
}

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
	status = sgx_ecall(eid, 2, &ocall_table_Enclave, &ms);
	return status;
}

