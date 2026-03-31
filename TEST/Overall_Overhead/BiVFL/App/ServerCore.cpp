/* App/ServerCore.cpp */
#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <iostream>
#include <cstring>
#include <openssl/sha.h> 

typedef unsigned __int128 uint128;
const long long MOD = 9223372036854775783;
const double SCALE = 100000000.0; 

// [极速优化] 移植无分支的 uint128 运算
class MathUtils {
public:
    static inline long long safe_mod_add(long long a, long long b) {
        uint128 res = (uint128)a + (uint128)b;
        return (long long)(res % MOD);
    }
    static inline long long safe_mod_sub(long long a, long long b) {
        long long res = a - b;
        if (res < 0) res += MOD;
        return res;
    }
    static inline long long safe_mod_mul(long long a, long long b) {
        uint128 res = (uint128)a * (uint128)b;
        return (long long)(res % MOD);
    }
    static long long mod_inverse(long long n) {
        if (n == 0) return 0;
        long long base = n; if (base < 0) base += MOD;
        uint128 exp = MOD - 2; 
        uint128 res = 1; 
        uint128 b = base % MOD;
        while (exp > 0) {
            if (exp % 2 == 1) res = (res * b) % MOD;
            b = (b * b) % MOD;
            exp /= 2;
        }
        return (long long)res;
    }
};

class CryptoUtils {
public:
    static long derive_seed(long root, const char* purpose, int id) {
        std::string s = std::to_string(root) + "_" + purpose + "_" + std::to_string(id);
        unsigned char hash[SHA256_DIGEST_LENGTH];
        SHA256((const unsigned char*)s.c_str(), s.length(), hash);
        uint32_t seed_val;
        std::memcpy(&seed_val, hash, sizeof(uint32_t));
        return (long)(seed_val & 0x7FFFFFFF);
    }
};

// [极速优化] 移植现代流密码级别的 Xoshiro256** PRG
class DeterministicRandom {
private:
    uint64_t s[4];
    static inline uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }
    uint64_t next() {
        const uint64_t result = rotl(s[1] * 5, 7) * 9;
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t; s[3] = rotl(s[3], 45);
        return result;
    }
public:
    DeterministicRandom(long seed) {
        s[0] = (uint64_t)seed ^ 0x1234567890ABCDEFULL;
        s[1] = (uint64_t)seed ^ 0xFEDCBA0987654321ULL;
        s[2] = (uint64_t)seed ^ 0x13579BDF2468ACE0ULL;
        s[3] = (uint64_t)seed ^ 0x0ECA8642FDB97531ULL;
        for(int i=0; i<16; i++) next(); // 预热打乱
    }
    inline long long next_mask_mod() {
        return (long long)(next() % MOD); 
    }
};

long long lagrange_interpolate_zero(const std::vector<int>& xs, const std::vector<long long>& ys) {
    long long result = 0; int k = xs.size();
    for (int i = 0; i < k; ++i) {
        long long num = 1; long long den = 1;
        for (int j = 0; j < k; ++j) {
            if (i != j) {
                long long neg_xj = MathUtils::safe_mod_sub(0, xs[j]);
                num = MathUtils::safe_mod_mul(num, neg_xj);
                long long diff = MathUtils::safe_mod_sub(xs[i], xs[j]);
                den = MathUtils::safe_mod_mul(den, diff);
            }
        }
        long long term = MathUtils::safe_mod_mul(ys[i], num);
        long long inv_den = MathUtils::mod_inverse(den);
        term = MathUtils::safe_mod_mul(term, inv_den);
        result = MathUtils::safe_mod_add(result, term);
    }
    return result;
}

extern "C" {
void ta_offline_compute(
    int* u1_ids, int u1_len, const char* kappa_m_str, int t, int param_size,
    long long* out_s_alpha, long long* out_s_beta
) {
    long kappa_m = std::strtol(kappa_m_str, NULL, 10);
    std::vector<DeterministicRandom> ta_alpha_rngs, ta_beta_rngs;
    for (int i = 0; i < u1_len; ++i) {
        ta_alpha_rngs.emplace_back(CryptoUtils::derive_seed(kappa_m, "alpha", u1_ids[i] * 1000 + t));
        ta_beta_rngs.emplace_back(CryptoUtils::derive_seed(kappa_m, "beta",  u1_ids[i] * 1000 + t));
    }

    for (int k = 0; k < param_size; ++k) {
        long long cur_S_alpha = 0; long long cur_S_beta = 0;
        for(auto& rng : ta_alpha_rngs) cur_S_alpha = MathUtils::safe_mod_add(cur_S_alpha, rng.next_mask_mod());
        for(auto& rng : ta_beta_rngs)  cur_S_beta  = MathUtils::safe_mod_add(cur_S_beta,  rng.next_mask_mod());
        out_s_alpha[k] = cur_S_alpha;
        out_s_beta[k]  = cur_S_beta;
    }
}

void server_core_aggregate_and_unmask(
    int* u1_ids, int u1_len, int* u2_ids, int u2_len, 
    long long* shares_flat, int shares_len, long long* ciphers_flat, int data_len, 
    const char* kappa_m_str, int t, const char* model_hash_str, int threshold, 
    const long long* ta_s_alpha, const long long* ta_s_beta, 
    float* output_result 
) {
    long kappa_m = std::strtol(kappa_m_str, NULL, 10);
    long long H_val = std::stoll(model_hash_str) % MOD;

    std::map<int, std::vector<std::pair<int, long long>>> shares_self, shares_alpha, shares_beta;

    for (int i = 0; i < shares_len; i += 4) {
        int source = shares_flat[i]; int tag = shares_flat[i+1];
        int target = shares_flat[i+2]; long long val = shares_flat[i+3];
        int x_coord = source + 1; 
        if (tag == 1) shares_self[target].push_back({x_coord, val});
        else if (tag == 2) shares_alpha[target].push_back({x_coord, val});
        else if (tag == 3) shares_beta[target].push_back({x_coord, val});
    }

    auto interpolate = [&](int target, const std::vector<std::pair<int, long long>>& pts) -> long long {
        if((int)pts.size() < threshold) return 0; 
        std::vector<int> xs; std::vector<long long> ys;
        for(int k=0; k<threshold; ++k) { xs.push_back(pts[k].first); ys.push_back(pts[k].second); }
        return lagrange_interpolate_zero(xs, ys);
    };

    std::vector<DeterministicRandom> active_self_rngs, dropped_alpha_rngs, dropped_beta_rngs;
    
    for(auto& kv : shares_self) { long long s = interpolate(kv.first, kv.second); if(s > 0) active_self_rngs.emplace_back((long)s); }
    for(auto& kv : shares_alpha) { long long s = interpolate(kv.first, kv.second); if(s > 0) dropped_alpha_rngs.emplace_back((long)s); }
    for(auto& kv : shares_beta) { long long s = interpolate(kv.first, kv.second); if(s > 0) dropped_beta_rngs.emplace_back((long)s); }

    for (int k = 0; k < data_len; ++k) {
        long long sum_cipher = 0;
        for(int i=0; i<u2_len; ++i) sum_cipher = MathUtils::safe_mod_add(sum_cipher, ciphers_flat[i * data_len + k]);

        long long cur_S_alpha = ta_s_alpha[k];
        long long cur_S_beta  = ta_s_beta[k];

        for(auto& rng : dropped_alpha_rngs) cur_S_alpha = MathUtils::safe_mod_sub(cur_S_alpha, rng.next_mask_mod());
        for(auto& rng : dropped_beta_rngs)  cur_S_beta  = MathUtils::safe_mod_sub(cur_S_beta,  rng.next_mask_mod());

        long long active_self_sum = 0;
        for(auto& rng : active_self_rngs) active_self_sum = MathUtils::safe_mod_add(active_self_sum, rng.next_mask_mod());

        long long term_alpha = MathUtils::safe_mod_mul(cur_S_alpha, H_val);
        long long noise = MathUtils::safe_mod_add(term_alpha, cur_S_beta);
        noise = MathUtils::safe_mod_add(noise, active_self_sum);

        long long res = MathUtils::safe_mod_sub(sum_cipher, noise);
        if (res > MOD / 2) res -= MOD;
        output_result[k] = (float)((double)res / SCALE);
    }
}
}