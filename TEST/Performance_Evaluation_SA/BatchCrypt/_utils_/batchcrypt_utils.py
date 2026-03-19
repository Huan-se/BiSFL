import os
import numpy as np

# 导入您刚刚上传的核心库
from .batchcrypt_core.paillier import PaillierKeypair
from .batchcrypt_core import encryption as bc_enc

class BatchCryptUtils:
    def __init__(self, num_clients, bit_width=16):
        self.bit_width = bit_width
        # 为防止多客户端聚合时加法溢出，必须预留 pad_zero
        # 100 个客户端相加最大需要 log2(100) ≈ 7 个 pad bit
        self.pad_zero = 7
        # Here's 3 is original coder's hard code, but it's not enough for 100 or more clients. We just want to make sure the testing of communication volume and time is correct, but the result is not right at all.
        
        # Paillier 默认是 2048 位的 n，每个 batch_size = 2048 // (bit_width + pad_zero)
        # 为了保守起见，留一些余量
        self.batch_size = 2000 // (self.bit_width + self.pad_zero)
        
    @staticmethod
    def generate_keypair(key_length=2048):
        """生成 Paillier 公私钥对 (耗时操作，仅在初始化时执行)"""
        pub_key, priv_key = PaillierKeypair.generate_keypair(n_length=key_length)
        return pub_key, priv_key

    def encrypt_gradients(self, public_key, gradients):
        """客户端：量化 -> 打包 -> Paillier 加密"""
        # 1. 使用 ACIQ 计算最优裁剪阈值 (r_max)
        r_max = bc_enc.calculate_clip_threshold_aciq_g([gradients], [gradients.size], self.bit_width)[0]
        r_max = r_max * 10000
        # 2. 调用原库函数：打包并加密
        # 注意：原库的 encrypt_matrix_batch 返回的是 (加密数组, 原形状)
        encrypted_batch, og_shape = bc_enc.encrypt_matrix_batch(
            public_key, 
            gradients, 
            batch_size=self.batch_size, 
            bit_width=self.bit_width, 
            pad_zero=self.pad_zero, 
            r_max=r_max
        )
        return encrypted_batch, og_shape, r_max

    @staticmethod
    def aggregate_ciphertexts(ciphertexts_list):
        """服务端：对收到的 Paillier 密文进行同态加法"""
        # Paillier 密文直接利用封装好的 __add__ 魔术方法相加即可
        aggregated = ciphertexts_list[0]
        for ct in ciphertexts_list[1:]:
            aggregated = aggregated + ct
        return aggregated

    def decrypt_and_unmask(self, private_key, aggregated_cipher, og_shape, r_max, active_count):
        """客户端：解密 -> 拆包 -> 反量化 -> 求平均"""
        # 调用原库函数解密并恢复成量化整数
        sum_gradients = bc_enc.decrypt_matrix_batch(
            private_key, 
            aggregated_cipher, 
            og_shape, 
            batch_size=self.batch_size, 
            bit_width=self.bit_width, 
            pad_zero=self.pad_zero, 
            r_max=r_max
        )
        # 求出平均明文梯度
        return sum_gradients / float(active_count)