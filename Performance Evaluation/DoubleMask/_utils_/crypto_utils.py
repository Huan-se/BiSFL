import os
import numpy as np
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

# 使用梅森素数 (2^127 - 1) 作为秘密共享的有限域 P，足以容纳我们的安全种子
PRIME = 2**127 - 1 

class CryptoUtils:
    
    # ==========================================
    # 1. 密钥协商 (Key Agreement)
    # ==========================================
    @staticmethod
    def generate_key_pair():
        """生成 X25519 椭圆曲线密钥对，返回 (私钥 bytes, 公钥 bytes)"""
        sk = x25519.X25519PrivateKey.generate()
        pk = sk.public_key()
        return sk.private_bytes_raw(), pk.public_bytes_raw()

    @staticmethod
    def agree(sk_bytes, pk_bytes):
        """计算 ECDH 共享密钥，并用 HKDF 导出纯净的 32 字节种子"""
        sk = x25519.X25519PrivateKey.from_private_bytes(sk_bytes)
        pk = x25519.X25519PublicKey.from_public_bytes(pk_bytes)
        shared_key = sk.exchange(pk)
        
        # 使用 HKDF 进行密钥派生，确保输出的随机性和安全性
        derived_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'secagg_key_agreement'
        ).derive(shared_key)
        return derived_key

    # ==========================================
    # 2. 认证加密 (Authenticated Encryption)
    # ==========================================
    @staticmethod
    def encrypt(key_32bytes, plaintext, associated_data=b""):
        """使用 AES-GCM 进行点对点消息的认证加密"""
        aesgcm = AESGCM(key_32bytes)
        nonce = os.urandom(12)  # NIST 推荐的 96-bit 随机 Nonce
        ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data)
        return nonce + ciphertext

    @staticmethod
    def decrypt(key_32bytes, ciphertext_with_nonce, associated_data=b""):
        """使用 AES-GCM 进行解密并验证完整性"""
        aesgcm = AESGCM(key_32bytes)
        nonce = ciphertext_with_nonce[:12]
        ciphertext = ciphertext_with_nonce[12:]
        return aesgcm.decrypt(nonce, ciphertext, associated_data)

    # ==========================================
    # 3. 秘密共享 (Shamir's Secret Sharing)
    # ==========================================
    @staticmethod
    def _eval_poly(poly, x):
        """在有限域 PRIME 上评估多项式 f(x)"""
        result = 0
        for coeff in reversed(poly):
            result = (result * x + coeff) % PRIME
        return result

    @staticmethod
    def share_secret(secret_int, t, n):
        """
        将整型秘密 secret_int 拆分为 n 份，恢复门限为 t
        返回字典: {用户ID(1~n): 分片值}
        """
        # 构造多项式: f(x) = a_0 + a_1*x + ... + a_{t-1}*x^{t-1}，其中 a_0 = secret
        poly = [secret_int] + [int.from_bytes(os.urandom(16), 'big') % PRIME for _ in range(t - 1)]
        shares = {}
        for x in range(1, n + 1):
            shares[x] = CryptoUtils._eval_poly(poly, x)
        return shares

    @staticmethod
    def reconstruct_secret(shares_dict):
        """
        利用拉格朗日插值法从大于等于 t 个分片中恢复秘密 (计算 f(0))
        shares_dict 格式: {用户ID_x: 分片值_y}
        """
        secret = 0
        shares = list(shares_dict.items())
        for i, (xi, yi) in enumerate(shares):
            num = 1
            den = 1
            for j, (xj, yj) in enumerate(shares):
                if i != j:
                    num = (num * (-xj)) % PRIME
                    den = (den * (xi - xj)) % PRIME
            # 使用费马小定理求分母在有限域的逆元
            inv_den = pow(den, PRIME - 2, PRIME)
            term = (yi * num * inv_den) % PRIME
            secret = (secret + term) % PRIME
        return secret

    # ==========================================
    # 4. 伪随机掩码生成器 (PRG)
    # ==========================================
    @staticmethod
    def generate_mask(seed_32bytes, size, mod=None):
        """
        利用 AES-CTR 将 32 字节种子极速扩展为大小为 size 的 int64 掩码数组
        """
        # 使用全零 IV，因为每次生成的 seed 已经是唯一的
        cipher = Cipher(algorithms.AES(seed_32bytes), modes.CTR(b'\x00' * 16))
        encryptor = cipher.encryptor()
        
        bytes_needed = size * 8  # numpy 的 int64 占用 8 字节
        zeros = np.zeros(bytes_needed, dtype=np.uint8).tobytes()
        rand_bytes = encryptor.update(zeros) + encryptor.finalize()
        
        # 将生成的伪随机字节流直接映射为 numpy 高维向量
        rand_array = np.frombuffer(rand_bytes, dtype=np.int64)
        if mod is not None:
            # Python 的 % 运算符能很好处理负数取模
            rand_array = rand_array % mod
        return rand_array

    # --- 辅助类型转换 ---
    @staticmethod
    def bytes_to_int(b):
        return int.from_bytes(b, 'big')

    @staticmethod
    def int_to_bytes(i, length=32):
        return i.to_bytes(length, 'big')