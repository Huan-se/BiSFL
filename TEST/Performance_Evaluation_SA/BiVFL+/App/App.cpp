#include "App.h"
#include "Enclave_u.h"
#include <sgx_urts.h>

// 【修复点】：显式定义并初始化 global_eid
sgx_enclave_id_t global_eid = 0;

extern "C" {
    void prepare_gradient(int client_id, int proj_seed, int param_size, 
                          const float* w_new, const float* w_old, float* out_proj) {
        ecall_prepare_gradient(global_eid, client_id, proj_seed, param_size, w_new, w_old, out_proj);
    }

    void generate_masked_gradient_sparse(
        uint32_t root_seed, uint32_t global_seed, int client_id,
        const int* active_neighbors, int num_neighbors,
        float weight, int param_size, int64_t* out_masked_gradient) {
        ecall_generate_masked_gradient_sparse(global_eid, root_seed, global_seed, client_id,
            active_neighbors, num_neighbors, weight, param_size, out_masked_gradient);
    }

    void get_scalar_shares_sparse(
        uint32_t seed_sss, uint32_t root_seed,
        const int* dropped_neighbors, int num_dropped,
        int client_id, int threshold, uint64_t* out_shares) {
        ecall_get_scalar_shares_sparse(global_eid, seed_sss, root_seed,
            dropped_neighbors, num_dropped, client_id, threshold, out_shares);
    }
}