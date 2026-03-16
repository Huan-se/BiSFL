#include <iostream>
#include <vector>
#include <random>
#include <chrono>
// 引入 Eigen 核心库
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main() {
    int param_size = 1000000;  // 100 万维参数
    int proj_size = 1024;      // 1024 维投影特征

    cout << "========================================" << endl;
    cout << "  C++ + Eigen 局部敏感哈希 (LSH) 性能测试" << endl;
    cout << "========================================" << endl;

    // 使用 Eigen 的 VectorXf 定义高维向量
    VectorXf w_new = VectorXf::Constant(param_size, 0.5f);
    VectorXf proj_result = VectorXf::Zero(proj_size);

    // SGX 内存受限 (EPC 仅 128MB)，不能直接分配 4GB 的二维矩阵。
    // 我们按行生成随机数并复用内存，这也符合原生 SGX Enclave 的真实场景。
    VectorXf random_row(param_size);

    mt19937 gen(12345);
    normal_distribution<float> dist(0.0f, 1.0f);

    double gen_time_total = 0.0;
    double dot_time_total = 0.0;

    auto start_time = chrono::high_resolution_clock::now();

    for (int i = 0; i < proj_size; ++i) {
        // [阶段 1] 顺序生成随机数
        auto t0 = chrono::high_resolution_clock::now();
        for (int j = 0; j < param_size; ++j) {
            random_row[j] = dist(gen);
        }
        auto t1 = chrono::high_resolution_clock::now();

        // [阶段 2] Eigen 向量点乘 (SIMD 极致加速)
        proj_result[i] = random_row.dot(w_new);
        auto t2 = chrono::high_resolution_clock::now();

        gen_time_total += chrono::duration<double>(t1 - t0).count();
        dot_time_total += chrono::duration<double>(t2 - t1).count();
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end_time - start_time;

    cout << "\n[Done] 投影计算完成!" << endl;
    cout << "打印第一个特征值验证: " << proj_result[0] << endl;
    cout << "========================================" << endl;
    cout << "📊 核心时间分布剖析:" << endl;
    cout << " -> [瓶颈] 随机数生成耗时: " << gen_time_total << " 秒" << endl;
    cout << " -> [加速] Eigen 点乘耗时: " << dot_time_total << " 秒" << endl;
    cout << "----------------------------------------" << endl;
    cout << "🔥 总计耗时: " << diff.count() << " 秒" << endl;
    cout << "========================================" << endl;

    return 0;
}