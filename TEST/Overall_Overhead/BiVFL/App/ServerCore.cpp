/* App/ServerCore.cpp */
#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <random>
#include <iostream>
#include <cstring>
#include <openssl/sha.h> 

typedef __int128_t int128;
const long long MOD = 9223372036854775783;
const double SCALE = 100000000.0; // 引入常量 SCALE

class MathUtils {
public:
    static long long safe_mod_add(long long a, long long b) {
        int128 ua = (int128)a; int128 ub = (int128)b;
        if (ua < 0) ua += MOD; if (ub < 0) ub += MOD;
        return (long long)((ua + ub) % (int128)MOD);
    }
    static long long safe_mod_sub(long long a, long long b) {
        int128 ua = (int128)a; int128 ub = (int128)b;
        if (ua < 0) ua += MOD; if (ub < 0) ub += MOD;
        int128 res = (ua - ub) % (int128)MOD;
        if (res < 0) res += MOD;
        return (long long)res;
    }
    static long long safe_mod_mul(long long a, long long b) {
        int128 ua = (int128)a; int128 ub = (int128)b;
        if (ua < 0) ua += MOD; if (ub < 0) ub += MOD;
        return (long long)((ua * ub) % (int128)MOD);
    }
    static long long mod_inverse(long long n) {
        if (n == 0) return 0;
        int128 base = (int128)n; if (base < 0) base += MOD;
        int128 exp = (int128)MOD - 2; 
        int128 res = 1; base %= MOD;
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
        std::string s = std::to_string(root) + "_" + purpose + "_" + std::to_string(id);
        unsigned char hash[SHA256_DIGEST_LENGTH];
        SHA256((const unsigned char*)s.c_str(), s.length(), hash);
        uint32_t seed_val;
        std::memcpy(&seed_val, hash, sizeof(uint32_t));
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
void server_core_aggregate_and_unmask(
    int* u1_ids, int u1_len, int* u2_ids, int u2_len, 
    long long* shares_flat, int shares_len, long long* ciphers_flat, int data_len, 
    const char* kappa_m_str, int t, const char* model_hash_str, int threshold, 
    float* output_result // [核心修改]：直接输出 float，避开 Python 截断
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

    std::vector<DeterministicRandom> ta_alpha_rngs, ta_beta_rngs;
    for (int i = 0; i < u1_len; ++i) {
        ta_alpha_rngs.emplace_back(CryptoUtils::derive_seed(kappa_m, "alpha", u1_ids[i] * 1000 + t));
        ta_beta_rngs.emplace_back(CryptoUtils::derive_seed(kappa_m, "beta",  u1_ids[i] * 1000 + t));
    }

    for (int k = 0; k < data_len; ++k) {
        long long sum_cipher = 0;
        for(int i=0; i<u2_len; ++i) sum_cipher = MathUtils::safe_mod_add(sum_cipher, ciphers_flat[i * data_len + k]);

        long long cur_S_alpha = 0; long long cur_S_beta  = 0;
        for(auto& rng : ta_alpha_rngs) cur_S_alpha = MathUtils::safe_mod_add(cur_S_alpha, rng.next_mask_mod());
        for(auto& rng : ta_beta_rngs)  cur_S_beta  = MathUtils::safe_mod_add(cur_S_beta,  rng.next_mask_mod());

        for(auto& rng : dropped_alpha_rngs) cur_S_alpha = MathUtils::safe_mod_sub(cur_S_alpha, rng.next_mask_mod());
        for(auto& rng : dropped_beta_rngs)  cur_S_beta  = MathUtils::safe_mod_sub(cur_S_beta,  rng.next_mask_mod());

        long long active_self_sum = 0;
        for(auto& rng : active_self_rngs) active_self_sum = MathUtils::safe_mod_add(active_self_sum, rng.next_mask_mod());

        long long term_alpha = MathUtils::safe_mod_mul(cur_S_alpha, H_val);
        long long noise = MathUtils::safe_mod_add(term_alpha, cur_S_beta);
        noise = MathUtils::safe_mod_add(noise, active_self_sum);

        // [核心修复]：由 C++ 在 int64 精度下完成完美减法和尺度还原
        long long res = MathUtils::safe_mod_sub(sum_cipher, noise);
        if (res > MOD / 2) {
            res -= MOD;
        }
        output_result[k] = (float)((double)res / SCALE);
    }
}
}