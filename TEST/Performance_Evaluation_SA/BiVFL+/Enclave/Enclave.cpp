#include "Enclave.h"
#include "Enclave_t.h"
#include <random>
#include <vector>

// 使用梅森素数 M31 作为秘密共享的有限域，极大提升计算速度且不会溢出 64 位寄存器
const uint64_t MODULUS = 2147483647;

void ecall_prepare_gradient(int client_id, int proj_seed, int param_size, const float* w_new, const float* w_old_cache, float* out_proj) {
    // 投影 Benchmark 桩函数，略
    for(int i=0; i<1024; i++) out_proj[i] = w_new[i % param_size];
}

void ecall_generate_masked_gradient_sparse(
    uint32_t root_seed, uint32_t global_seed, int client_id,
    const int* active_neighbors, int num_neighbors,
    float weight, int param_size, int64_t* out_masked_gradient)
{
    for(int i=0; i<param_size; i++) out_masked_gradient[i] = 0;

    // 1. 生成客户端自掩码 (Self Mask)
    uint32_t self_seed = root_seed ^ (client_id * 0x9E3779B9);
    std::mt19937 self_prg(self_seed);
    for(int i=0; i<param_size; i++) {
        out_masked_gradient[i] += (self_prg() % 10000);
    }

    // 2. O(log N) 稀疏成对掩码 (Pairwise Masks) 的确定性加减
    for(int k=0; k<num_neighbors; k++) {
        int neighbor_id = active_neighbors[k];
        int min_id = client_id < neighbor_id ? client_id : neighbor_id;
        int max_id = client_id > neighbor_id ? client_id : neighbor_id;
        
        uint32_t pair_seed = root_seed ^ (min_id * 0x85ebca6b) ^ (max_id * 0xc2b2ae35);
        std::mt19937 pair_prg(pair_seed);

        if (client_id > neighbor_id) {
            for(int i=0; i<param_size; i++) out_masked_gradient[i] += (pair_prg() % 10000);
        } else {
            for(int i=0; i<param_size; i++) out_masked_gradient[i] -= (pair_prg() % 10000);
        }
    }
}

void ecall_get_scalar_shares_sparse(
    uint32_t seed_sss, uint32_t root_seed,
    const int* dropped_neighbors, int num_dropped,
    int client_id, int threshold, uint64_t* out_shares)
{
    // 【核心创新落地】：零通信复现掉线者的多项式并求值
    for(int i=0; i<num_dropped; i++) {
        int dropped_id = dropped_neighbors[i];
        
        // 推导出掉线者的原始标量种子 b_k
        uint32_t dropped_seed = root_seed ^ (dropped_id * 0x9E3779B9);
        uint64_t b_k = dropped_seed % MODULUS;

        // 确定性生成掉线者的 Shamir 多项式系数 (f(x) = b_k + c_1*x + ... + c_{t-1}*x^{t-1})
        uint32_t poly_seed = seed_sss ^ (dropped_id * 0x1234567);
        std::mt19937 poly_prg(poly_seed);

        std::vector<uint64_t> coeffs(threshold);
        coeffs[0] = b_k; 
        for(int c=1; c<threshold; c++) {
            coeffs[c] = poly_prg() % MODULUS;
        }

        // 在本地代入 x = client_id 进行多项式求值，直接拿到自己的 Share！
        uint64_t x = client_id;
        uint64_t y = 0;
        uint64_t x_pow = 1;
        for(int c=0; c<threshold; c++) {
            uint64_t term = (coeffs[c] * x_pow) % MODULUS;
            y = (y + term) % MODULUS;
            x_pow = (x_pow * x) % MODULUS;
        }
        out_shares[i] = y;
    }
}