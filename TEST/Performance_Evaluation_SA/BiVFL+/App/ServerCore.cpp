#include <iostream>
#include <vector>
#include <map>
#include <random>

extern "C" {

const uint64_t MODULUS = 2147483647;

// 拓展欧几里得算法求模逆
uint64_t modInverse(uint64_t a, uint64_t m) {
    int64_t m0 = m, t, q;
    int64_t x0 = 0, x1 = 1;
    if (m == 1) return 0;
    while (a > 1) {
        q = a / m;
        t = m;
        m = a % m, a = t;
        t = x0;
        x0 = x1 - q * x0;
        x1 = t;
    }
    if (x1 < 0) x1 += m0;
    return x1;
}

void aggregate_and_unmask_sparse(
    uint32_t root_seed, uint32_t global_seed,
    const int* u1_ids, int num_u1,
    const int* u2_ids, int num_u2,
    const int* flat_owners, const uint64_t* flat_vals, int num_shares,
    const int64_t* ciphers, int num_ciphers, int param_size,
    int N_total, int K_degree,
    int64_t* out_plaintext)
{
    // 消除 unused parameter 编译警告
    (void)root_seed;
    (void)global_seed;
    (void)num_ciphers;
    (void)N_total;

    // 1. O(N*d) 极速汇聚所有幸存者的密文
    for(int i=0; i<param_size; i++) out_plaintext[i] = 0;
    for(int c=0; c<num_u2; c++) {
        const int64_t* client_cipher = ciphers + c * param_size;
        for(int i=0; i<param_size; i++) out_plaintext[i] += client_cipher[i];
    }

    // 2. 在 Server 端瞬间还原 K-正则稀疏图，自动对齐扁平化的标量分片
    std::vector<int> u1_vec(u1_ids, u1_ids + num_u1);
    std::map<int, std::vector<int>> graph;
    for (int idx = 0; idx < num_u1; idx++) {
        int cid = u1_vec[idx];
        for (int d = 1; d <= K_degree / 2; d++) {
            graph[cid].push_back(u1_vec[(idx + d) % num_u1]);
            graph[cid].push_back(u1_vec[(idx - d + num_u1) % num_u1]);
        }
    }

    std::map<int, std::vector<std::pair<uint64_t, uint64_t>>> share_groups; // 掉线者ID -> 收集到的 (x_i, y_i)
    int current_share_idx = 0;

    for(int j=0; j<num_u2; j++) {
        int provider_id = u2_ids[j];
        std::vector<int> dropped_neighbors;
        for(int n : graph[provider_id]) {
            bool dropped = true;
            for(int c=0; c<num_u2; c++) if(u2_ids[c] == n) { dropped = false; break; }
            if(dropped) dropped_neighbors.push_back(n);
        }
        // 将解包出的标量归档
        for(int k : dropped_neighbors) {
            if (current_share_idx < num_shares) {
                share_groups[k].push_back({ (uint64_t)provider_id, flat_vals[current_share_idx] });
                current_share_idx++;
            }
        }
    }

    // 3. Lagrange 标量插值还原与 PRG O(d) 膨胀消除
    size_t threshold = (size_t)(K_degree / 2 + 1);
    
    // 【C++11 兼容性修复】使用标准迭代器代替 C++17 的 structured bindings
    for (std::map<int, std::vector<std::pair<uint64_t, uint64_t>>>::const_iterator it = share_groups.begin(); it != share_groups.end(); ++it) {
        int dropped_id = it->first;
        (void)dropped_id; // 暂不需要，抑制警告
        const std::vector<std::pair<uint64_t, uint64_t>>& shares = it->second;

        if (shares.size() < threshold) continue; 

        uint64_t b_k = 0;
        // 计算 f(0) = \sum y_i \prod (-x_j) / (x_i - x_j)
        for (size_t i = 0; i < threshold; i++) {
            uint64_t x_i = shares[i].first;
            uint64_t y_i = shares[i].second;

            uint64_t num = 1;
            uint64_t den = 1;
            for (size_t j = 0; j < threshold; j++) {
                if (i == j) continue;
                uint64_t x_j = shares[j].first;

                uint64_t neg_x_j = (MODULUS - (x_j % MODULUS)) % MODULUS;
                num = (num * neg_x_j) % MODULUS;

                uint64_t diff = (x_i + MODULUS - (x_j % MODULUS)) % MODULUS;
                den = (den * diff) % MODULUS;
            }

            uint64_t den_inv = modInverse(den, MODULUS);
            uint64_t term = (y_i * num) % MODULUS;
            term = (term * den_inv) % MODULUS;
            b_k = (b_k + term) % MODULUS;
        }

        // PRG 膨胀：将 32 位标量爆开成 D 维度掩码，消除成对残余
        std::mt19937 prg(b_k);
        for(int i=0; i<param_size; i++) {
            out_plaintext[i] -= (prg() % 10000); 
        }
    }
}
}