import os
import numpy as np
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

# 使用 2^63 - 25 作为秘密共享的有限域 P
PRIME = 9223372036854775783 

class CryptoUtils:
    
    # ==========================================
    # 1. 密钥协商 (Key Agreement)
    # ==========================================
    @staticmethod
    def generate_key_pair():
        sk = x25519.X25519PrivateKey.generate()
        pk = sk.public_key()
        return sk.private_bytes_raw(), pk.public_bytes_raw()

    @staticmethod
    def agree(sk_bytes, pk_bytes):
        sk = x25519.X25519PrivateKey.from_private_bytes(sk_bytes)
        pk = x25519.X25519PublicKey.from_public_bytes(pk_bytes)
        shared_key = sk.exchange(pk)
        
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
        aesgcm = AESGCM(key_32bytes)
        nonce = os.urandom(12) 
        ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data)
        return nonce + ciphertext

    @staticmethod
    def decrypt(key_32bytes, ciphertext_with_nonce, associated_data=b""):
        aesgcm = AESGCM(key_32bytes)
        nonce = ciphertext_with_nonce[:12]
        ciphertext = ciphertext_with_nonce[12:]
        return aesgcm.decrypt(nonce, ciphertext, associated_data)

    # ==========================================
    # 3. 秘密共享 (Shamir's Secret Sharing)
    # ==========================================
    @staticmethod
    def _eval_poly(poly, x):
        result = 0
        for coeff in reversed(poly):
            result = (result * x + coeff) % PRIME
        return result

    @staticmethod
    def share_secret(secret_int, t, user_ids):
        """
        将整型秘密 secret_int 拆分，恢复门限为 t
        只在 user_ids 指定的 x 坐标上生成分片，适配稀疏图拓扑
        """
        # 8 bytes 的随机数足以覆盖 2^63 - 25 的范围
        poly = [secret_int] + [int.from_bytes(os.urandom(8), 'big') % PRIME for _ in range(t - 1)]
        shares = {}
        for x in user_ids:
            shares[x] = CryptoUtils._eval_poly(poly, x)
        return shares

    @staticmethod
    def reconstruct_secret(shares_dict):
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
    # 4. 伪随机掩码生成器 (PRG)
    # ==========================================
    @staticmethod
    def generate_mask(seed_32bytes, size, mod=None):
        cipher = Cipher(algorithms.AES(seed_32bytes), modes.CTR(b'\x00' * 16))
        encryptor = cipher.encryptor()
        
        bytes_needed = size * 8
        zeros = np.zeros(bytes_needed, dtype=np.uint8).tobytes()
        rand_bytes = encryptor.update(zeros) + encryptor.finalize()
        
        rand_array = np.frombuffer(rand_bytes, dtype=np.int64)
        if mod is not None:
            rand_array = rand_array % mod
        return rand_array

    @staticmethod
    def bytes_to_int(b):
        return int.from_bytes(b, 'big')

    @staticmethod
    def int_to_bytes(i, length=32):
        return i.to_bytes(length, 'big')