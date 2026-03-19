import os
import numpy as np
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import serialization

# 论文指定：秘密共享与加扰使用的素数域 2^63 - 25
PRIME = 2**63 - 25

# 用于大整数切片的常量 (60 bits，确保绝对小于 PRIME)
CHUNK_BITS = 60
CHUNK_MASK = (1 << CHUNK_BITS) - 1

class CryptoUtils:
    
    # ==========================================
    # 1. 密钥协商: ECDH (NIST P-256) + SHA-256
    # ==========================================
    @staticmethod
    def generate_key_pair():
        """生成 NIST P-256 椭圆曲线密钥对，返回 (私钥int, 公钥bytes)"""
        sk = ec.generate_private_key(ec.SECP256R1())
        # 提取私钥整数，方便后续进行 Shamir 秘密共享
        sk_int = sk.private_numbers().private_value
        pk_bytes = sk.public_key().public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.CompressedPoint
        )
        return sk_int, pk_bytes

    @staticmethod
    def agree(sk_int, pk_bytes, length=16):
        """计算 ECDH 共享密钥，使用 SHA-256 派生 (默认16字节用于 AES-128)"""
        sk = ec.derive_private_key(sk_int, ec.SECP256R1())
        pk = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), pk_bytes)
        shared_key = sk.exchange(ec.ECDH(), pk)
        
        # 组合 SHA-256 哈希函数进行 HKDF 密钥派生
        derived_key = HKDF(
            algorithm=hashes.SHA256(),
            length=length,
            salt=None,
            info=b'secagg_key_agreement'
        ).derive(shared_key)
        return derived_key

    # ==========================================
    # 2. 认证加密: AES-GCM (128-bit keys)
    # ==========================================
    @staticmethod
    def encrypt(key_16bytes, plaintext, associated_data=b""):
        aesgcm = AESGCM(key_16bytes)
        nonce = os.urandom(12) 
        ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data)
        return nonce + ciphertext

    @staticmethod
    def decrypt(key_16bytes, ciphertext_with_nonce, associated_data=b""):
        aesgcm = AESGCM(key_16bytes)
        nonce = ciphertext_with_nonce[:12]
        ciphertext = ciphertext_with_nonce[12:]
        return aesgcm.decrypt(nonce, ciphertext, associated_data)

    # ==========================================
    # 3. 秘密共享: t-out-of-n Shamir Sharing (域: 2^63-25)
    # ==========================================
    @staticmethod
    def _eval_poly(poly, x):
        result = 0
        for coeff in reversed(poly):
            result = (result * x + coeff) % PRIME
        return result

    @staticmethod
    def share_secret(secret_int, t, n):
        """支持将大整数（如256位）切片为 60-bit 在 2^63-25 域上共享"""
        chunks = []
        temp = secret_int
        if temp == 0:
            chunks.append(0)
        while temp > 0:
            chunks.append(temp & CHUNK_MASK)
            temp >>= CHUNK_BITS

        shares = {i: [] for i in range(1, n + 1)}
        for chunk in chunks:
            # 随机生成系数，最高位必须在 PRIME 范围内
            poly = [chunk] + [int.from_bytes(os.urandom(8), 'big') % PRIME for _ in range(t - 1)]
            for x in range(1, n + 1):
                shares[x].append(CryptoUtils._eval_poly(poly, x))
        return shares

    @staticmethod
    def reconstruct_secret(shares_dict):
        """恢复大整数秘密"""
        uids = list(shares_dict.keys())
        num_chunks = len(shares_dict[uids[0]])

        secret_int = 0
        for chunk_idx in range(num_chunks):
            chunk_shares = {uid: shares_dict[uid][chunk_idx] for uid in uids}
            chunk_val = CryptoUtils._reconstruct_single(chunk_shares)
            secret_int |= (chunk_val << (CHUNK_BITS * chunk_idx))
        return secret_int

    @staticmethod
    def _reconstruct_single(shares_dict):
        secret = 0
        shares = list(shares_dict.items())
        for i, (xi, yi) in enumerate(shares):
            num = 1
            den = 1
            for j, (xj, yj) in enumerate(shares):
                if i != j:
                    num = (num * (-xj)) % PRIME
                    den = (den * (xi - xj)) % PRIME
            inv_den = pow(den, PRIME - 2, PRIME)
            term = (yi * num * inv_den) % PRIME
            secret = (secret + term) % PRIME
        return secret

    # ==========================================
    # 4. PRG: AES in counter (CTR) mode
    # ==========================================
    @staticmethod
    def generate_mask(seed_16bytes, size, mod=None):
        cipher = Cipher(algorithms.AES(seed_16bytes), modes.CTR(b'\x00' * 16))
        encryptor = cipher.encryptor()
        
        bytes_needed = size * 8  
        zeros = np.zeros(bytes_needed, dtype=np.uint8).tobytes()
        rand_bytes = encryptor.update(zeros) + encryptor.finalize()
        
        # 使用 uint64 映射，完美适配 2^63-25 的模运算而不会负数溢出
        rand_array = np.frombuffer(rand_bytes, dtype=np.uint64)
        if mod is not None:
            rand_array = rand_array % np.uint64(mod)
        return rand_array.astype(np.int64)