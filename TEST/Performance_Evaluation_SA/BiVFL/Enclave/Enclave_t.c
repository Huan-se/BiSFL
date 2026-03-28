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
	int _tmp_param_size = ms->ms_param_size;
	size_t _len_w_new = _tmp_param_size * sizeof(float);
	float* _in_w_new = NULL;
	const float* _tmp_w_old_cache = ms->ms_w_old_cache;
	size_t _len_w_old_cache = _tmp_param_size * sizeof(float);
	float* _in_w_old_cache = NULL;
	float* _tmp_out_proj = ms->ms_out_proj;
	size_t _len_out_proj = 1024 * sizeof(float);
	float* _in_out_proj = NULL;

	if (sizeof(*_tmp_w_new) != 0 &&
		(size_t)_tmp_param_size > (SIZE_MAX / sizeof(*_tmp_w_new))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	if (sizeof(*_tmp_w_old_cache) != 0 &&
		(size_t)_tmp_param_size > (SIZE_MAX / sizeof(*_tmp_w_old_cache))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	if (sizeof(*_tmp_out_proj) != 0 &&
		1024 > (SIZE_MAX / sizeof(*_tmp_out_proj))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	CHECK_UNIQUE_POINTER(_tmp_w_new, _len_w_new);
	CHECK_UNIQUE_POINTER(_tmp_w_old_cache, _len_w_old_cache);
	CHECK_UNIQUE_POINTER(_tmp_out_proj, _len_out_proj);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp_w_new != NULL && _len_w_new != 0) {
		if ( _len_w_new % sizeof(*_tmp_w_new) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_w_new = (float*)malloc(_len_w_new);
		if (_in_w_new == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_w_new, _len_w_new, _tmp_w_new, _len_w_new)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}
	if (_tmp_w_old_cache != NULL && _len_w_old_cache != 0) {
		if ( _len_w_old_cache % sizeof(*_tmp_w_old_cache) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_w_old_cache = (float*)malloc(_len_w_old_cache);
		if (_in_w_old_cache == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_w_old_cache, _len_w_old_cache, _tmp_w_old_cache, _len_w_old_cache)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}
	if (_tmp_out_proj != NULL && _len_out_proj != 0) {
		if ( _len_out_proj % sizeof(*_tmp_out_proj) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		if ((_in_out_proj = (float*)malloc(_len_out_proj)) == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_out_proj, 0, _len_out_proj);
	}

	ecall_prepare_gradient(ms->ms_client_id, ms->ms_proj_seed, _tmp_param_size, (const float*)_in_w_new, (const float*)_in_w_old_cache, _in_out_proj);
	if (_in_out_proj) {
		if (memcpy_s(_tmp_out_proj, _len_out_proj, _in_out_proj, _len_out_proj)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}

err:
	if (_in_w_new) free(_in_w_new);
	if (_in_w_old_cache) free(_in_w_old_cache);
	if (_in_out_proj) free(_in_out_proj);
	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_generate_masked_gradient_sparse(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_generate_masked_gradient_sparse_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_generate_masked_gradient_sparse_t* ms = SGX_CAST(ms_ecall_generate_masked_gradient_sparse_t*, pms);
	sgx_status_t status = SGX_SUCCESS;
	const char* _tmp_kappa_m_str = ms->ms_kappa_m_str;
	size_t _len_kappa_m_str = ms->ms_kappa_m_str_len ;
	char* _in_kappa_m_str = NULL;
	const char* _tmp_model_hash_str = ms->ms_model_hash_str;
	size_t _len_model_hash_str = ms->ms_model_hash_str_len ;
	char* _in_model_hash_str = NULL;
	const float* _tmp_w_new = ms->ms_w_new;
	int _tmp_param_size = ms->ms_param_size;
	size_t _len_w_new = _tmp_param_size * sizeof(float);
	float* _in_w_new = NULL;
	int64_t* _tmp_out_masked_gradient = ms->ms_out_masked_gradient;
	size_t _len_out_masked_gradient = _tmp_param_size * sizeof(int64_t);
	int64_t* _in_out_masked_gradient = NULL;

	if (sizeof(*_tmp_w_new) != 0 &&
		(size_t)_tmp_param_size > (SIZE_MAX / sizeof(*_tmp_w_new))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	if (sizeof(*_tmp_out_masked_gradient) != 0 &&
		(size_t)_tmp_param_size > (SIZE_MAX / sizeof(*_tmp_out_masked_gradient))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	CHECK_UNIQUE_POINTER(_tmp_kappa_m_str, _len_kappa_m_str);
	CHECK_UNIQUE_POINTER(_tmp_model_hash_str, _len_model_hash_str);
	CHECK_UNIQUE_POINTER(_tmp_w_new, _len_w_new);
	CHECK_UNIQUE_POINTER(_tmp_out_masked_gradient, _len_out_masked_gradient);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp_kappa_m_str != NULL && _len_kappa_m_str != 0) {
		_in_kappa_m_str = (char*)malloc(_len_kappa_m_str);
		if (_in_kappa_m_str == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_kappa_m_str, _len_kappa_m_str, _tmp_kappa_m_str, _len_kappa_m_str)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

		_in_kappa_m_str[_len_kappa_m_str - 1] = '\0';
		if (_len_kappa_m_str != strlen(_in_kappa_m_str) + 1)
		{
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}
	if (_tmp_model_hash_str != NULL && _len_model_hash_str != 0) {
		_in_model_hash_str = (char*)malloc(_len_model_hash_str);
		if (_in_model_hash_str == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_model_hash_str, _len_model_hash_str, _tmp_model_hash_str, _len_model_hash_str)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

		_in_model_hash_str[_len_model_hash_str - 1] = '\0';
		if (_len_model_hash_str != strlen(_in_model_hash_str) + 1)
		{
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}
	if (_tmp_w_new != NULL && _len_w_new != 0) {
		if ( _len_w_new % sizeof(*_tmp_w_new) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_w_new = (float*)malloc(_len_w_new);
		if (_in_w_new == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_w_new, _len_w_new, _tmp_w_new, _len_w_new)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}
	if (_tmp_out_masked_gradient != NULL && _len_out_masked_gradient != 0) {
		if ( _len_out_masked_gradient % sizeof(*_tmp_out_masked_gradient) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		if ((_in_out_masked_gradient = (int64_t*)malloc(_len_out_masked_gradient)) == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_out_masked_gradient, 0, _len_out_masked_gradient);
	}

	ecall_generate_masked_gradient_sparse((const char*)_in_kappa_m_str, ms->ms_t, (const char*)_in_model_hash_str, ms->ms_client_id, (const float*)_in_w_new, ms->ms_weight, _tmp_param_size, _in_out_masked_gradient);
	if (_in_out_masked_gradient) {
		if (memcpy_s(_tmp_out_masked_gradient, _len_out_masked_gradient, _in_out_masked_gradient, _len_out_masked_gradient)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}

err:
	if (_in_kappa_m_str) free(_in_kappa_m_str);
	if (_in_model_hash_str) free(_in_model_hash_str);
	if (_in_w_new) free(_in_w_new);
	if (_in_out_masked_gradient) free(_in_out_masked_gradient);
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
	const char* _tmp_kappa_s_str = ms->ms_kappa_s_str;
	size_t _len_kappa_s_str = ms->ms_kappa_s_str_len ;
	char* _in_kappa_s_str = NULL;
	const char* _tmp_kappa_m_str = ms->ms_kappa_m_str;
	size_t _len_kappa_m_str = ms->ms_kappa_m_str_len ;
	char* _in_kappa_m_str = NULL;
	const char* _tmp_view_hash_str = ms->ms_view_hash_str;
	size_t _len_view_hash_str = ms->ms_view_hash_str_len ;
	char* _in_view_hash_str = NULL;
	const int* _tmp_alive_neighbors = ms->ms_alive_neighbors;
	int _tmp_num_alive = ms->ms_num_alive;
	size_t _len_alive_neighbors = _tmp_num_alive * sizeof(int);
	int* _in_alive_neighbors = NULL;
	const int* _tmp_dropped_neighbors = ms->ms_dropped_neighbors;
	int _tmp_num_dropped = ms->ms_num_dropped;
	size_t _len_dropped_neighbors = _tmp_num_dropped * sizeof(int);
	int* _in_dropped_neighbors = NULL;
	int64_t* _tmp_out_shares = ms->ms_out_shares;
	size_t _tmp_max_len = ms->ms_max_len;
	size_t _len_out_shares = _tmp_max_len * sizeof(int64_t);
	int64_t* _in_out_shares = NULL;

	if (sizeof(*_tmp_alive_neighbors) != 0 &&
		(size_t)_tmp_num_alive > (SIZE_MAX / sizeof(*_tmp_alive_neighbors))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	if (sizeof(*_tmp_dropped_neighbors) != 0 &&
		(size_t)_tmp_num_dropped > (SIZE_MAX / sizeof(*_tmp_dropped_neighbors))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	if (sizeof(*_tmp_out_shares) != 0 &&
		(size_t)_tmp_max_len > (SIZE_MAX / sizeof(*_tmp_out_shares))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	CHECK_UNIQUE_POINTER(_tmp_kappa_s_str, _len_kappa_s_str);
	CHECK_UNIQUE_POINTER(_tmp_kappa_m_str, _len_kappa_m_str);
	CHECK_UNIQUE_POINTER(_tmp_view_hash_str, _len_view_hash_str);
	CHECK_UNIQUE_POINTER(_tmp_alive_neighbors, _len_alive_neighbors);
	CHECK_UNIQUE_POINTER(_tmp_dropped_neighbors, _len_dropped_neighbors);
	CHECK_UNIQUE_POINTER(_tmp_out_shares, _len_out_shares);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp_kappa_s_str != NULL && _len_kappa_s_str != 0) {
		_in_kappa_s_str = (char*)malloc(_len_kappa_s_str);
		if (_in_kappa_s_str == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_kappa_s_str, _len_kappa_s_str, _tmp_kappa_s_str, _len_kappa_s_str)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

		_in_kappa_s_str[_len_kappa_s_str - 1] = '\0';
		if (_len_kappa_s_str != strlen(_in_kappa_s_str) + 1)
		{
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}
	if (_tmp_kappa_m_str != NULL && _len_kappa_m_str != 0) {
		_in_kappa_m_str = (char*)malloc(_len_kappa_m_str);
		if (_in_kappa_m_str == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_kappa_m_str, _len_kappa_m_str, _tmp_kappa_m_str, _len_kappa_m_str)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

		_in_kappa_m_str[_len_kappa_m_str - 1] = '\0';
		if (_len_kappa_m_str != strlen(_in_kappa_m_str) + 1)
		{
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}
	if (_tmp_view_hash_str != NULL && _len_view_hash_str != 0) {
		_in_view_hash_str = (char*)malloc(_len_view_hash_str);
		if (_in_view_hash_str == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_view_hash_str, _len_view_hash_str, _tmp_view_hash_str, _len_view_hash_str)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

		_in_view_hash_str[_len_view_hash_str - 1] = '\0';
		if (_len_view_hash_str != strlen(_in_view_hash_str) + 1)
		{
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}
	if (_tmp_alive_neighbors != NULL && _len_alive_neighbors != 0) {
		if ( _len_alive_neighbors % sizeof(*_tmp_alive_neighbors) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_alive_neighbors = (int*)malloc(_len_alive_neighbors);
		if (_in_alive_neighbors == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_alive_neighbors, _len_alive_neighbors, _tmp_alive_neighbors, _len_alive_neighbors)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}
	if (_tmp_dropped_neighbors != NULL && _len_dropped_neighbors != 0) {
		if ( _len_dropped_neighbors % sizeof(*_tmp_dropped_neighbors) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_dropped_neighbors = (int*)malloc(_len_dropped_neighbors);
		if (_in_dropped_neighbors == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_dropped_neighbors, _len_dropped_neighbors, _tmp_dropped_neighbors, _len_dropped_neighbors)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}
	if (_tmp_out_shares != NULL && _len_out_shares != 0) {
		if ( _len_out_shares % sizeof(*_tmp_out_shares) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		if ((_in_out_shares = (int64_t*)malloc(_len_out_shares)) == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		memset((void*)_in_out_shares, 0, _len_out_shares);
	}

	ecall_get_scalar_shares_sparse((const char*)_in_kappa_s_str, (const char*)_in_kappa_m_str, ms->ms_t, (const char*)_in_view_hash_str, ms->ms_client_id, (const int*)_in_alive_neighbors, _tmp_num_alive, (const int*)_in_dropped_neighbors, _tmp_num_dropped, ms->ms_threshold, _in_out_shares, _tmp_max_len);
	if (_in_out_shares) {
		if (memcpy_s(_tmp_out_shares, _len_out_shares, _in_out_shares, _len_out_shares)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}
	}

err:
	if (_in_kappa_s_str) free(_in_kappa_s_str);
	if (_in_kappa_m_str) free(_in_kappa_m_str);
	if (_in_view_hash_str) free(_in_view_hash_str);
	if (_in_alive_neighbors) free(_in_alive_neighbors);
	if (_in_dropped_neighbors) free(_in_dropped_neighbors);
	if (_in_out_shares) free(_in_out_shares);
	return status;
}

SGX_EXTERNC const struct {
	size_t nr_ecall;
	struct {void* ecall_addr; uint8_t is_priv; uint8_t is_switchless;} ecall_table[3];
} g_ecall_table = {
	3,
	{
		{(void*)(uintptr_t)sgx_ecall_prepare_gradient, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_generate_masked_gradient_sparse, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_get_scalar_shares_sparse, 0, 0},
	}
};

SGX_EXTERNC const struct {
	size_t nr_ocall;
	uint8_t entry_table[6][3];
} g_dyn_entry_table = {
	6,
	{
		{0, 0, 0, },
		{0, 0, 0, },
		{0, 0, 0, },
		{0, 0, 0, },
		{0, 0, 0, },
		{0, 0, 0, },
	}
};


sgx_status_t SGX_CDECL ocall_print_string(const char* str)
{
	sgx_status_t status = SGX_SUCCESS;
	size_t _len_str = str ? strlen(str) + 1 : 0;

	ms_ocall_print_string_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_ocall_print_string_t);
	void *__tmp = NULL;


	CHECK_ENCLAVE_POINTER(str, _len_str);

	if (ADD_ASSIGN_OVERFLOW(ocalloc_size, (str != NULL) ? _len_str : 0))
		return SGX_ERROR_INVALID_PARAMETER;

	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_ocall_print_string_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_ocall_print_string_t));
	ocalloc_size -= sizeof(ms_ocall_print_string_t);

	if (str != NULL) {
		ms->ms_str = (const char*)__tmp;
		if (_len_str % sizeof(*str) != 0) {
			sgx_ocfree();
			return SGX_ERROR_INVALID_PARAMETER;
		}
		if (memcpy_s(__tmp, ocalloc_size, str, _len_str)) {
			sgx_ocfree();
			return SGX_ERROR_UNEXPECTED;
		}
		__tmp = (void *)((size_t)__tmp + _len_str);
		ocalloc_size -= _len_str;
	} else {
		ms->ms_str = NULL;
	}
	
	status = sgx_ocall(0, ms);

	if (status == SGX_SUCCESS) {
	}
	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL sgx_oc_cpuidex(int cpuinfo[4], int leaf, int subleaf)
{
	sgx_status_t status = SGX_SUCCESS;
	size_t _len_cpuinfo = 4 * sizeof(int);

	ms_sgx_oc_cpuidex_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_sgx_oc_cpuidex_t);
	void *__tmp = NULL;

	void *__tmp_cpuinfo = NULL;

	CHECK_ENCLAVE_POINTER(cpuinfo, _len_cpuinfo);

	if (ADD_ASSIGN_OVERFLOW(ocalloc_size, (cpuinfo != NULL) ? _len_cpuinfo : 0))
		return SGX_ERROR_INVALID_PARAMETER;

	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_sgx_oc_cpuidex_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_sgx_oc_cpuidex_t));
	ocalloc_size -= sizeof(ms_sgx_oc_cpuidex_t);

	if (cpuinfo != NULL) {
		ms->ms_cpuinfo = (int*)__tmp;
		__tmp_cpuinfo = __tmp;
		if (_len_cpuinfo % sizeof(*cpuinfo) != 0) {
			sgx_ocfree();
			return SGX_ERROR_INVALID_PARAMETER;
		}
		memset(__tmp_cpuinfo, 0, _len_cpuinfo);
		__tmp = (void *)((size_t)__tmp + _len_cpuinfo);
		ocalloc_size -= _len_cpuinfo;
	} else {
		ms->ms_cpuinfo = NULL;
	}
	
	ms->ms_leaf = leaf;
	ms->ms_subleaf = subleaf;
	status = sgx_ocall(1, ms);

	if (status == SGX_SUCCESS) {
		if (cpuinfo) {
			if (memcpy_s((void*)cpuinfo, _len_cpuinfo, __tmp_cpuinfo, _len_cpuinfo)) {
				sgx_ocfree();
				return SGX_ERROR_UNEXPECTED;
			}
		}
	}
	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL sgx_thread_wait_untrusted_event_ocall(int* retval, const void* self)
{
	sgx_status_t status = SGX_SUCCESS;

	ms_sgx_thread_wait_untrusted_event_ocall_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_sgx_thread_wait_untrusted_event_ocall_t);
	void *__tmp = NULL;


	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_sgx_thread_wait_untrusted_event_ocall_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_sgx_thread_wait_untrusted_event_ocall_t));
	ocalloc_size -= sizeof(ms_sgx_thread_wait_untrusted_event_ocall_t);

	ms->ms_self = self;
	status = sgx_ocall(2, ms);

	if (status == SGX_SUCCESS) {
		if (retval) *retval = ms->ms_retval;
	}
	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL sgx_thread_set_untrusted_event_ocall(int* retval, const void* waiter)
{
	sgx_status_t status = SGX_SUCCESS;

	ms_sgx_thread_set_untrusted_event_ocall_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_sgx_thread_set_untrusted_event_ocall_t);
	void *__tmp = NULL;


	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_sgx_thread_set_untrusted_event_ocall_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_sgx_thread_set_untrusted_event_ocall_t));
	ocalloc_size -= sizeof(ms_sgx_thread_set_untrusted_event_ocall_t);

	ms->ms_waiter = waiter;
	status = sgx_ocall(3, ms);

	if (status == SGX_SUCCESS) {
		if (retval) *retval = ms->ms_retval;
	}
	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL sgx_thread_setwait_untrusted_events_ocall(int* retval, const void* waiter, const void* self)
{
	sgx_status_t status = SGX_SUCCESS;

	ms_sgx_thread_setwait_untrusted_events_ocall_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_sgx_thread_setwait_untrusted_events_ocall_t);
	void *__tmp = NULL;


	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_sgx_thread_setwait_untrusted_events_ocall_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_sgx_thread_setwait_untrusted_events_ocall_t));
	ocalloc_size -= sizeof(ms_sgx_thread_setwait_untrusted_events_ocall_t);

	ms->ms_waiter = waiter;
	ms->ms_self = self;
	status = sgx_ocall(4, ms);

	if (status == SGX_SUCCESS) {
		if (retval) *retval = ms->ms_retval;
	}
	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL sgx_thread_set_multiple_untrusted_events_ocall(int* retval, const void** waiters, size_t total)
{
	sgx_status_t status = SGX_SUCCESS;
	size_t _len_waiters = total * sizeof(void*);

	ms_sgx_thread_set_multiple_untrusted_events_ocall_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_sgx_thread_set_multiple_untrusted_events_ocall_t);
	void *__tmp = NULL;


	CHECK_ENCLAVE_POINTER(waiters, _len_waiters);

	if (ADD_ASSIGN_OVERFLOW(ocalloc_size, (waiters != NULL) ? _len_waiters : 0))
		return SGX_ERROR_INVALID_PARAMETER;

	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_sgx_thread_set_multiple_untrusted_events_ocall_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_sgx_thread_set_multiple_untrusted_events_ocall_t));
	ocalloc_size -= sizeof(ms_sgx_thread_set_multiple_untrusted_events_ocall_t);

	if (waiters != NULL) {
		ms->ms_waiters = (const void**)__tmp;
		if (_len_waiters % sizeof(*waiters) != 0) {
			sgx_ocfree();
			return SGX_ERROR_INVALID_PARAMETER;
		}
		if (memcpy_s(__tmp, ocalloc_size, waiters, _len_waiters)) {
			sgx_ocfree();
			return SGX_ERROR_UNEXPECTED;
		}
		__tmp = (void *)((size_t)__tmp + _len_waiters);
		ocalloc_size -= _len_waiters;
	} else {
		ms->ms_waiters = NULL;
	}
	
	ms->ms_total = total;
	status = sgx_ocall(5, ms);

	if (status == SGX_SUCCESS) {
		if (retval) *retval = ms->ms_retval;
	}
	sgx_ocfree();
	return status;
}

