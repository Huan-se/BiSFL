#include "Enclave_t.h"
#include <random>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include "sgx_tcrypto.h"

typedef __int128_t int128;
const long long MOD = 9223372036854775783;
const double SCALE = 100000000.0;

long parse_long(const char* str) { if (!str) return 0; char* end; return std::strtol(str, &end, 10); }
long long parse_long_long(const char* str) { if (!str) return 0; return std::stoll(str); }

class MathUtils {
public:
    static long long safe_mod_add(long long a, long long b) {
        int128 ua = (int128)a; int128 ub = (int128)b;
        if (ua < 0) ua += MOD; if (ub < 0) ub += MOD;
        return (long long)((ua + ub) % (int128)MOD);
    }
    static long long safe_mod_mul(long long a, long long b) {
        int128 ua = (int128)a; int128 ub = (int128)b;
        if (ua < 0) ua += MOD; if (ub < 0) ub += MOD;
        return (long long)((ua * ub) % (int128)MOD);
    }
};

class CryptoUtils {
public:
    static long derive_seed(long root, const char* purpose, int id) {
        std::string s = std::to_string(root) + "_" + purpose + "_" + std::to_string(id);
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
        do { val = ((uint64_t)gen() << 32) | gen(); } while (val >= limit);
        return (long long)(val % MOD);
    }
};

void ecall_prepare_gradient(int client_id, int proj_seed, int param_size, const float* w_new, const float* w_old_cache, float* out_proj) {
    for(int i=0; i<1024; i++) out_proj[i] = w_new[i % param_size];
}

void ecall_generate_masked_gradient_sparse(
    const char* kappa_m_str, int t, const char* model_hash_str,
    int client_id, const float* w_new, float weight, int param_size, int64_t* out_masked_gradient)
{
    long kappa_m = parse_long(kappa_m_str);
    long long H_val = parse_long_long(model_hash_str) % MOD;

    long s_alpha = CryptoUtils::derive_seed(kappa_m, "alpha", client_id * 1000 + t);
    long s_beta  = CryptoUtils::derive_seed(kappa_m, "beta",  client_id * 1000 + t);
    long s_self  = CryptoUtils::derive_seed(kappa_m, "self",  client_id * 1000 + t);
    
    DeterministicRandom rng_A(s_alpha), rng_B(s_beta), rng_S(s_self);

    for(int i=0; i<param_size; i++) {
        float g = w_new[i]; 
        long long G = (long long)(g * weight * SCALE);
        G = (G % MOD + MOD) % MOD;

        long long val_A = rng_A.next_mask_mod();
        long long term_A = MathUtils::safe_mod_mul(val_A, H_val);
        long long val_B = rng_B.next_mask_mod();
        long long val_S = rng_S.next_mask_mod();

        long long C = MathUtils::safe_mod_add(G, term_A);
        C = MathUtils::safe_mod_add(C, val_B);
        C = MathUtils::safe_mod_add(C, val_S);
        
        out_masked_gradient[i] = C;
    }
}

void ecall_get_scalar_shares_sparse(
    const char* kappa_s_str, const char* kappa_m_str, int t, const char* view_hash_str,
    int client_id, const int* alive_neighbors, int num_alive,
    const int* dropped_neighbors, int num_dropped, int threshold,
    int64_t* out_shares, size_t max_len)
{
    long kappa_s = parse_long(kappa_s_str);
    long kappa_m = parse_long(kappa_m_str);
    long long view_hash = parse_long_long(view_hash_str);

    int out_idx = 1; 
    int count = 0;
    long long x_val = client_id + 1; 

    auto generate_share = [&](int tag, int target, long seed) {
        if(out_idx + 2 >= max_len) return;
        std::string purpose = std::to_string(tag) + "_" + std::to_string(target) + "_" + std::to_string(view_hash);
        long poly_seed = CryptoUtils::derive_seed(kappa_s, purpose.c_str(), t);
        DeterministicRandom rng_poly(poly_seed);
        
        long long res = seed; 
        long long x_pow = x_val;
        for (int i = 1; i < threshold; ++i) {
            long long coeff = rng_poly.next_mask_mod();
            long long term = MathUtils::safe_mod_mul(coeff, x_pow);
            res = MathUtils::safe_mod_add(res, term);
            x_pow = MathUtils::safe_mod_mul(x_pow, x_val);
        }
        out_shares[out_idx++] = tag;
        out_shares[out_idx++] = target;
        out_shares[out_idx++] = res;
        count++;
    };

    for(int i=0; i<num_alive; i++) {
        int a = alive_neighbors[i];
        generate_share(1, a, CryptoUtils::derive_seed(kappa_m, "self", a * 1000 + t));
    }
    for(int i=0; i<num_dropped; i++) {
        int d = dropped_neighbors[i];
        generate_share(2, d, CryptoUtils::derive_seed(kappa_m, "alpha", d * 1000 + t));
        generate_share(3, d, CryptoUtils::derive_seed(kappa_m, "beta",  d * 1000 + t));
    }
    out_shares[0] = count;
}