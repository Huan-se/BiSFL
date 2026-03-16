#include "Enclave_t.h"

#include "sgx_trts.h" /* for sgx_ocalloc, sgx_is_outside_enclave */
#include "sgx_lfence.h" /* for sgx_lfence */

#include <errno.h>
#include <mbusafecrt.h> /* for memcpy_s etc */
#include <stdlib.h> /* for malloc/free etc */

#define CHECK_REF_POINTER(ptr, siz) do {	\
	if (!(ptr) || ! sgx_is_outside_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)

#define CHECK_UNIQUE_POINTER(ptr, siz) do {	\
	if ((ptr) && ! sgx_is_outside_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)

#define CHECK_ENCLAVE_POINTER(ptr, siz) do {	\
	if ((ptr) && ! sgx_is_within_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)

#define ADD_ASSIGN_OVERFLOW(a, b) (	\
	((a) += (b)) < (b)	\
)


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

static sgx_status_t SGX_CDECL sgx_ecall_generate_masked_gradient_sparse(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_generate_masked_gradient_sparse_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_generate_masked_gradient_sparse_t* ms = SGX_CAST(ms_ecall_generate_masked_gradient_sparse_t*, pms);
	sgx_status_t status = SGX_SUCCESS;
	const int* _tmp_active_neighbors = ms->ms_active_neighbors;
	int64_t* _tmp_out_masked_gradient = ms->ms_out_masked_gradient;



	ecall_generate_masked_gradient_sparse(ms->ms_root_seed, ms->ms_global_seed, ms->ms_client_id, (const int*)_tmp_active_neighbors, ms->ms_num_neighbors, ms->ms_weight, ms->ms_param_size, _tmp_out_masked_gradient);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_get_scalar_shares_sparse(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_get_scalar_shares_sparse_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_get_scalar_shares_sparse_t* ms = SGX_CAST(ms_ecall_get_scalar_shares_sparse_t*, pms);
	sgx_status_t status = SGX_SUCCESS;
	const int* _tmp_dropped_neighbors = ms->ms_dropped_neighbors;
	uint64_t* _tmp_out_shares = ms->ms_out_shares;



	ecall_get_scalar_shares_sparse(ms->ms_seed_sss, ms->ms_root_seed, (const int*)_tmp_dropped_neighbors, ms->ms_num_dropped, ms->ms_client_id, ms->ms_threshold, _tmp_out_shares);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_prepare_gradient(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_prepare_gradient_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_prepare_gradient_t* ms = SGX_CAST(ms_ecall_prepare_gradient_t*, pms);
	sgx_status_t status = SGX_SUCCESS;
	const float* _tmp_w_new = ms->ms_w_new;
	const float* _tmp_w_old_cache = ms->ms_w_old_cache;
	float* _tmp_out_proj = ms->ms_out_proj;



	ecall_prepare_gradient(ms->ms_client_id, ms->ms_proj_seed, ms->ms_param_size, (const float*)_tmp_w_new, (const float*)_tmp_w_old_cache, _tmp_out_proj);


	return status;
}

SGX_EXTERNC const struct {
	size_t nr_ecall;
	struct {void* ecall_addr; uint8_t is_priv; uint8_t is_switchless;} ecall_table[3];
} g_ecall_table = {
	3,
	{
		{(void*)(uintptr_t)sgx_ecall_generate_masked_gradient_sparse, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_get_scalar_shares_sparse, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_prepare_gradient, 0, 0},
	}
};

SGX_EXTERNC const struct {
	size_t nr_ocall;
} g_dyn_entry_table = {
	0,
};


