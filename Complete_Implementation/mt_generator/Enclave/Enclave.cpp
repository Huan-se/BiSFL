/* Enclave/Enclave.cpp */
#include "Enclave_t.h"
#include "sgx_trts.h"
#include "sgx_tcrypto.h"
#include "sgx_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

static int g_enclave_verbose = 0;

// SGX 内部全局变量，用于保存 RATLS 状态和最终解密出的种子
sgx_ecc_state_handle_t ecc_state = NULL;
sgx_ec256_private_t enclave_priv_key;
sgx_ec256_public_t enclave_pub_key;
long global_seed_mask_root = 0;

void ecall_ra_keygen(uint8_t* out_pub_key, uint8_t* out_quote) {
    sgx_ecc256_open_context(&ecc_state);
    sgx_ecc256_create_key_pair(&enclave_priv_key, &enclave_pub_key, ecc_state);
    memcpy(out_pub_key, &enclave_pub_key, sizeof(sgx_ec256_public_t));
    sgx_read_rand(out_quote, 4384); 
}

void ecall_ra_provision_seed(uint8_t* server_pub_key, uint8_t* cipher_payload) {
    sgx_ec256_public_t srv_pub;
    memcpy(&srv_pub, server_pub_key, sizeof(sgx_ec256_public_t));
    sgx_ec256_dh_shared_t shared_key;
    sgx_ecc256_compute_shared_dhkey(&enclave_priv_key, &srv_pub, &shared_key, ecc_state);
    sgx_ecc256_close_context(ecc_state);
    memset(&enclave_priv_key, 0, sizeof(sgx_ec256_private_t));
}

#define LOG_DEBUG(fmt, ...) \
    do { if (g_enclave_verbose) printf("[Enclave DEBUG] " fmt, ##__VA_ARGS__); } while (0)

void ecall_set_verbose(int level) {
    g_enclave_verbose = level;
}

extern "C" {
    int rand(void);
    void srand(unsigned int seed);
}
#ifndef RAND_MAX
#define RAND_MAX 2147483647
#endif

namespace std {
    using ::rand;
    using ::srand;
}

int printf(const char *fmt, ...) {
    char buf[BUFSIZ] = { '\0' };
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_print_string(buf);
    return 0;
}

#include <string>
#include <vector>
#include <map>
#include <random>
#include <cmath>
#include <algorithm> 
#include <mutex>
#include <Eigen/Dense>

typedef __int128_t int128;

#define CHUNK_SIZE 4096
const long long MOD = 9223372036854775783;
const double SCALE = 100000000.0; 
const uint64_t N_MASK = 0xFFFFFFFFFFFF; 

static std::map<int, std::vector<float>> g_gradient_buffer;
static std::mutex g_map_mutex;

long parse_long(const char* str) {
    if (!str) return 0;
    char* end;
    return std::strtol(str, &end, 10);
}

long long parse_long_long(const char* str) {
    if (!str) return 0;
    return std::stoll(str);
}

float parse_float(const char* str) {
    if (!str) return 0.0f;
    char* end;
    return std::strtof(str, &end);
}

class MathUtils {
public:
    static long long safe_mod_add(long long a, long long b) {
        int128 ua = (int128)a;
        int128 ub = (int128)b;
        if (ua < 0) ua += MOD;
        if (ub < 0) ub += MOD;
        return (long long)((ua + ub) % (int128)MOD);
    }

    static long long safe_mod_sub(long long a, long long b) {
        int128 ua = (int128)a;
        int128 ub = (int128)b;
        if (ua < 0) ua += MOD;
        if (ub < 0) ub += MOD;
        int128 res = (ua - ub) % (int128)MOD;
        if (res < 0) res += MOD;
        return (long long)res;
    }

    static long long safe_mod_mul(long long a, long long b) {
        int128 ua = (int128)a;
        int128 ub = (int128)b;
        if (ua < 0) ua += MOD;
        if (ub < 0) ub += MOD;
        return (long long)((ua * ub) % (int128)MOD);
    }

    static long long mod_inverse(long long n) {
        if (n == 0) return 0;
        int128 base = (int128)n;
        if (base < 0) base += MOD;
        int128 exp = (int128)MOD - 2; 
        int128 res = 1;
        base %= MOD;
        while (exp > 0) {
            if (exp % 2 == 1) res = (res * base) % MOD;
            base = (base * base) % MOD;
            exp /= 2;
        }
        return (long long)res;
    }
};

class CryptoUtils {
public:
    static long derive_seed(long root, const char* purpose, int id) {
        std::string s = std::to_string(root) + purpose + std::to_string(id);
        sgx_sha256_hash_t hash_output;
        sgx_sha256_msg((const uint8_t*)s.c_str(), (uint32_t)s.length(), &hash_output);
        uint32_t seed_val;
        memcpy(&seed_val, hash_output, sizeof(uint32_t));
        return (long)(seed_val & 0x7FFFFFFF);
    }
};

class DeterministicRandom {
private: std::mt19937 gen;
public:
    DeterministicRandom(long seed) : gen((unsigned int)seed) {}
    
    long long next_mask_mod() { 
        uint64_t limit = UINT64_MAX - (UINT64_MAX % MOD);
        uint64_t val;
        do {
            val = ((uint64_t)gen() << 32) | gen();
        } while (val >= limit);
        return (long long)(val % MOD);
    }
};

class FastAESRandom {
private:
    sgx_aes_ctr_128bit_key_t key;
    uint8_t ctr[16];

public:
    FastAESRandom(long seed) {
        memset(&key, 0, sizeof(key));
        memset(ctr, 0, sizeof(ctr));
        memcpy(&key, &seed, sizeof(long));
    }

    void generate_rademacher_chunk(Eigen::VectorXf& chunk, int size) {
        std::vector<uint8_t> zeros(size, 0); 
        std::vector<uint8_t> rands(size, 0);
        sgx_aes_ctr_encrypt(&key, zeros.data(), zeros.size(), ctr, 128, rands.data());

        for (int i = 0; i < size; ++i) {
            chunk[i] = (rands[i] & 1) ? 1.0f : -1.0f;
        }
    }
};

void GenerateGraph(int n, long kappa_m, int t, std::vector<int>& u1_ids, int my_id, 
                   std::vector<int>& out_neighbors, int& out_tau) {
    if (n < 2) return;
    
    int K = 3 * std::log2(n) + 9;
    int required_K = 2 * std::ceil(0.5 * n) + 2; 
    if (K < required_K) K = required_K;
    if (K >= n) K = n - 1;
    if (K % 2 != 0) K--; 
    
    out_tau = K - std::ceil(0.5 * n);
    if (out_tau < 2) out_tau = 2;

    long rho = CryptoUtils::derive_seed(kappa_m, "graph", t);
    std::vector<int> pi(n);
    for(int i=0; i<n; ++i) pi[i] = i;
    
    for(int l = n - 1; l >= 1; --l) {
        long s_l = CryptoUtils::derive_seed(rho, "shuffle", l);
        DeterministicRandom rng(s_l);
        int r_l = rng.next_mask_mod() % (l + 1);
        std::swap(pi[l], pi[r_l]);
    }

    int my_idx = -1;
    for(int i=0; i<n; ++i) {
        if(u1_ids[i] == my_id) { my_idx = i; break; }
    }
    if(my_idx == -1) return;

    int my_logical = -1;
    for(int i=0; i<n; ++i) {
        if(pi[i] == my_idx) { my_logical = i; break; }
    }

    for(int d = 1; d <= K / 2; ++d) {
        int neighbor_forward = (my_logical + d) % n;
        int neighbor_backward = (my_logical - d + n) % n;
        out_neighbors.push_back(u1_ids[pi[neighbor_forward]]);
        out_neighbors.push_back(u1_ids[pi[neighbor_backward]]);
    }
}


extern "C" {

void ecall_prepare_gradient(
    int client_id, const char* proj_seed_str,
    float* w_new, float* w_old, size_t model_len, 
    int* ranges, size_t ranges_len, float* output_proj, size_t out_len
) {
    long proj_seed = parse_long(proj_seed_str);
    try {
        std::vector<float> full_gradient;
        full_gradient.reserve(model_len);
        for(size_t i = 0; i < model_len; ++i) {
            full_gradient.push_back(w_new[i] - w_old[i]);
        }
        
        {
            std::lock_guard<std::mutex> lock(g_map_mutex);
            g_gradient_buffer[client_id] = full_gradient;
        }

        FastAESRandom fast_rng(proj_seed);
        Eigen::VectorXf rng_chunk(CHUNK_SIZE);

        for (size_t k = 0; k < out_len; ++k) {
            float dot_product = 0.0f;
            
            for (size_t r = 0; r < ranges_len; r += 2) {
                int start_idx = ranges[r];
                int block_len = ranges[r+1];
                if (start_idx < 0 || start_idx + block_len > (int)model_len) continue;
                
                int offset = 0;
                while (offset < block_len) {
                    int curr_size = std::min((int)CHUNK_SIZE, block_len - offset);
                    
                    fast_rng.generate_rademacher_chunk(rng_chunk, curr_size);

                    Eigen::Map<Eigen::VectorXf> grad_segment(
                        full_gradient.data() + start_idx + offset, 
                        curr_size
                    );
                    dot_product += rng_chunk.head(curr_size).dot(grad_segment);
                    
                    offset += curr_size;
                }
            }
            output_proj[k] = dot_product;
        }

    } catch (...) {
        LOG_DEBUG("[Enclave Error] OOM or Exception in prepare_gradient!\n");
        for(size_t i=0; i<out_len; ++i) output_proj[i] = 0.0f;
    }
}

void ecall_generate_masked_gradient_dynamic(
    const char* kappa_m_str, int t, const char* model_hash_str, int client_id,
    int* u1_ids, size_t u1_len, const char* k_weight_str, size_t model_len, 
    int* ranges, size_t ranges_len, long long* output, size_t out_len
) {
    long kappa_m = parse_long(kappa_m_str);
    long long H_val = parse_long_long(model_hash_str) % MOD;
    float k_weight = parse_float(k_weight_str);

    std::vector<float> grad;
    {
        std::lock_guard<std::mutex> lock(g_map_mutex);
        if(g_gradient_buffer.count(client_id) == 0) return;
        grad = g_gradient_buffer[client_id];
    }

    long s_alpha = CryptoUtils::derive_seed(kappa_m, "alpha", client_id * 1000 + t);
    long s_beta  = CryptoUtils::derive_seed(kappa_m, "beta",  client_id * 1000 + t);
    long s_self  = CryptoUtils::derive_seed(kappa_m, "self",  client_id * 1000 + t);
    
    DeterministicRandom rng_A(s_alpha);
    DeterministicRandom rng_B(s_beta);
    DeterministicRandom rng_S(s_self);

    size_t cur = 0;
    for (size_t r = 0; r < ranges_len; r += 2) {
        int start = ranges[r]; int len = ranges[r+1];
        for(int i=0; i<len && cur < out_len; ++i) {
            float g = grad[start+i];
            long long G = (long long)(g * k_weight * SCALE);
            G = (G % MOD + MOD) % MOD;

            long long val_A = rng_A.next_mask_mod();
            long long term_A = MathUtils::safe_mod_mul(val_A, H_val);
            long long val_B = rng_B.next_mask_mod();
            long long val_S = rng_S.next_mask_mod();

            long long C = MathUtils::safe_mod_add(G, term_A);
            C = MathUtils::safe_mod_add(C, val_B);
            C = MathUtils::safe_mod_add(C, val_S);
            
            output[cur++] = C;
        }
    }
}

void ecall_get_vector_shares_dynamic(
    const char* kappa_s_str, const char* kappa_m_str, int t,
    int* u1_ids, size_t u1_len, int* u2_ids, size_t u2_len, 
    int my_id, long long* output_vector, size_t max_len
) {
    long kappa_s = parse_long(kappa_s_str);
    long kappa_m = parse_long(kappa_m_str);

    if (u2_len < std::ceil(0.5 * u1_len)) return; 

    std::vector<int> vec_u1(u1_ids, u1_ids + u1_len);
    std::vector<int> vec_u2(u2_ids, u2_ids + u2_len);
    
    std::vector<int> neighbors;
    int tau;
    GenerateGraph(u1_len, kappa_m, t, vec_u1, my_id, neighbors, tau);

    std::vector<int> alive_neighbors;
    std::vector<int> dropped_neighbors;
    for(int nb : neighbors) {
        if(std::find(vec_u2.begin(), vec_u2.end(), nb) != vec_u2.end()) alive_neighbors.push_back(nb);
        else dropped_neighbors.push_back(nb);
    }

    if (alive_neighbors.size() < (size_t)tau) return; 

    long long view_hash = 0;
    for(int a : vec_u2) view_hash = MathUtils::safe_mod_add(view_hash, a); 

    int out_idx = 1; 
    int count = 0;
    long long x_val = my_id + 1; 

    auto generate_share = [&](int tag, int target, long seed) {
        if(out_idx + 2 >= max_len) return;
        long poly_seed = CryptoUtils::derive_seed(kappa_s, std::to_string(tag + target + view_hash).c_str(), t);
        DeterministicRandom rng_poly(poly_seed);
        
        long long res = seed; 
        long long x_pow = x_val;
        for (int i = 1; i < tau; ++i) {
            long long coeff = rng_poly.next_mask_mod();
            long long term = MathUtils::safe_mod_mul(coeff, x_pow);
            res = MathUtils::safe_mod_add(res, term);
            x_pow = MathUtils::safe_mod_mul(x_pow, x_val);
        }
        output_vector[out_idx++] = tag;
        output_vector[out_idx++] = target;
        output_vector[out_idx++] = res;
        count++;
    };

    for(int a : alive_neighbors) {
        long s_self = CryptoUtils::derive_seed(kappa_m, "self", a * 1000 + t);
        generate_share(1, a, s_self);
    }

    for(int d : dropped_neighbors) {
        long s_alpha = CryptoUtils::derive_seed(kappa_m, "alpha", d * 1000 + t);
        long s_beta  = CryptoUtils::derive_seed(kappa_m, "beta",  d * 1000 + t);
        generate_share(2, d, s_alpha);
        generate_share(3, d, s_beta);
    }

    output_vector[0] = count;
}

void ecall_generate_noise_from_seed(const char* seed_str, size_t len, long long* output) {
    long seed = parse_long(seed_str);
    try {
        DeterministicRandom rng(seed);
        for(size_t i=0; i<len; ++i) output[i] = rng.next_mask_mod();
    } catch (...) {}
}

} // extern "C"