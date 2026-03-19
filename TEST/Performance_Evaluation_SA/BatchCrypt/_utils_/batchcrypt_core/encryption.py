import datetime
import tensorflow as tf
import numpy as np
from numba import njit, prange
import math
import random
from .paillier import PaillierPublicKey, PaillierPrivateKey
from . import aciq

import multiprocessing
from joblib import Parallel, delayed

import warnings

N_JOBS = multiprocessing.cpu_count()


def encrypt(public_key: PaillierPublicKey, x):
    return public_key.encrypt(x)


def encrypt_array(public_key: PaillierPublicKey, A):
    encrypt_A = Parallel(n_jobs=N_JOBS)(delayed(public_key.encrypt)(num) for num in A)
    return np.array(encrypt_A)


def encrypt_matrix(public_key: PaillierPublicKey, A):
    og_shape = A.shape
    if len(A.shape) == 1:
        A = np.expand_dims(A, axis=0)

    A = np.reshape(A, (1, -1))
    A = np.squeeze(A)
    encrypt_A = Parallel(n_jobs=N_JOBS)(delayed(public_key.encrypt)(num) for num in A)
    encrypt_A = np.expand_dims(encrypt_A, axis=0)
    encrypt_A = np.reshape(encrypt_A, og_shape)
    return np.array(encrypt_A)


@njit(parallel=True)
def stochastic_r(ori, frac, rand):
    result = np.zeros(len(ori), dtype=np.int32)
    for i in prange(len(ori)):
        if frac[i] >= 0:
            result[i] = np.floor(ori[i]) if frac[i] <= rand[i] else np.ceil(ori[i])
        else:
            result[i] = np.floor(ori[i]) if (-1 * frac[i]) > rand[i] else np.ceil(ori[i])
    return result


def stochastic_round(ori):
    rand = np.random.rand(len(ori))
    frac, decim = np.modf(ori)
    result = stochastic_r(ori, frac, rand)
    return result.astype(int)


def stochastic_round_matrix(ori):
    _shape = ori.shape
    ori = np.reshape(ori, (1, -1))
    ori = np.squeeze(ori)
    result = stochastic_round(ori)
    result = result.reshape(_shape)
    return result


def quantize_matrix(matrix, bit_width=8, r_max=0.5):
    og_sign = np.sign(matrix)
    uns_matrix = matrix * og_sign
    uns_result = (uns_matrix * (pow(2, bit_width - 1) - 1.0) / r_max)
    result = (og_sign * uns_result)
    return result, og_sign


def quantize_matrix_stochastic(matrix, bit_width=8, r_max=0.5):
    og_sign = np.sign(matrix)
    uns_matrix = matrix * og_sign
    uns_result = (uns_matrix * (pow(2, bit_width - 1) - 1.0) / r_max)
    result = (og_sign * uns_result)
    result = stochastic_round_matrix(result)
    return result, og_sign


def unquantize_matrix(matrix, bit_width=8, r_max=0.5):
    matrix = matrix.astype(int)
    og_sign = np.sign(matrix)
    uns_matrix = matrix * og_sign
    uns_result = uns_matrix * r_max / (pow(2, bit_width - 1) - 1.0)
    result = og_sign * uns_result
    return result.astype(np.float32)


def true_to_two_comp(input, bit_width):
    def true_to_two(value, bit_width):
        if value < 0:
            return 2 ** (bit_width + 1) + value
        else:
            return value
    result = Parallel(n_jobs=N_JOBS)(delayed(true_to_two)(x, bit_width) for x in input)
    return np.array(result)


@njit(parallel=True)
def true_to_two_comp_(input, bit_width):
    result = np.zeros(len(input), dtype=np.int32)
    for i in prange(len(input)):
        if input[i] >= 0:
            result[i] = input[i]
        else:
            result[i] = 2 ** (bit_width + 1) + input[i]
    return result


def encrypt_matrix_batch(public_key: PaillierPublicKey, A, batch_size=16, bit_width=8, pad_zero=3, r_max=0.5):
    og_shape = A.shape
    if len(A.shape) == 1:
        A = np.expand_dims(A, axis=0)

    A, og_sign = quantize_matrix(A, bit_width, r_max)

    A = np.reshape(A, (1, -1))
    A = np.squeeze(A)
    A = stochastic_round(A)

    A_len = len(A)
    A = A if (A_len % batch_size) == 0 \
        else np.pad(A, (0, batch_size - (A_len % batch_size)), 'constant', constant_values=(0, 0))

    A = true_to_two_comp_(A, bit_width)

    idx_range = int(len(A) / batch_size)
    idx_base = list(range(idx_range))
    batched_nums = np.array([pow(2, 2048)] * idx_range)
    batched_nums *= 0
    for i in range(batch_size):
        idx_filter = [i + x * batch_size for x in idx_base]
        filted_num = A[idx_filter]
        batched_nums = (batched_nums * pow(2, (bit_width + pad_zero))) + filted_num

    encrypt_A = Parallel(n_jobs=N_JOBS)(delayed(public_key.encrypt)(num) for num in batched_nums)
    return encrypt_A, og_shape


def encrypt_matmul(public_key: PaillierPublicKey, A, encrypted_B):
    if A.shape[-1] != encrypted_B.shape[0]:
        print("A and encrypted_B shape are not consistent")
        exit(1)
    res = [[public_key.encrypt(0) for _ in range(encrypted_B.shape[1])] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(encrypted_B.shape[1]):
            for m in range(len(A[i])):
                res[i][j] += A[i][m] * encrypted_B[m][j]
    return np.array(res)


def encrypt_matmul_3(public_key: PaillierPublicKey, A, encrypted_B):
    if A.shape[0] != encrypted_B.shape[0]:
        print("A and encrypted_B shape are not consistent")
        exit(1)
    res = []
    for i in range(len(A)):
        res.append(encrypt_matmul(public_key, A[i], encrypted_B[i]))
    return np.array(res)


def decrypt(private_key: PaillierPrivateKey, x):
    return private_key.decrypt(x)


def decrypt_scalar(private_key: PaillierPrivateKey, x):
    return private_key.decrypt(x)


def decrypt_array(private_key: PaillierPrivateKey, X):
    decrypt_x = []
    for i in range(X.shape[0]):
        elem = private_key.decrypt(X[i])
        decrypt_x.append(elem)
    return decrypt_x


def encrypt_array(private_key: PaillierPrivateKey, X):
    decrpt_X = Parallel(n_jobs=N_JOBS)(delayed(private_key.decrypt())(num) for num in X)
    return np.array(decrpt_X)


def decrypt_matrix(private_key: PaillierPrivateKey, A):
    og_shape = A.shape
    if len(A.shape) == 1:
        A = np.expand_dims(A, axis=0)

    A = np.reshape(A, (1, -1))
    A = np.squeeze(A)

    decrypt_A = Parallel(n_jobs=N_JOBS)(delayed(private_key.decrypt)(num) for num in A)

    decrypt_A = np.expand_dims(decrypt_A, axis=0)
    decrypt_A = np.reshape(decrypt_A, og_shape)

    return np.array(decrypt_A)


def two_comp_to_true(two_comp, bit_width=8, pad_zero=3):
    def binToInt(s, _bit_width=8):
        return int(s[1:], 2) - int(s[0]) * (1 << (_bit_width - 1))

    if two_comp < 0:
        raise Exception("Error: not expecting negtive value")
    two_com_string = bin(two_comp)[2:].zfill(bit_width + pad_zero)
    sign = two_com_string[0:pad_zero + 1]
    literal = two_com_string[pad_zero + 1:]

    if sign == '0' * (pad_zero + 1):  # positive value
        value = int(literal, 2)
        return value
    elif sign == '0' * (pad_zero - 2) + '1' + '0' * 2:  # positive value
        value = int(literal, 2)
        return value
    elif sign == '0' * pad_zero + '1':  # positive overflow
        value = pow(2, bit_width - 1) - 1
        return value
    elif sign == '0' * (pad_zero - 1) + '1' * 2:  # negtive value
        return binToInt('1' + literal, bit_width)
    elif sign == '0' * (pad_zero - 2) + '1' * 3:  # negtive value
        return binToInt('1' + literal, bit_width)
    elif sign == '0' * (pad_zero - 2) + '110':  # negtive overflow
        return - (pow(2, bit_width - 1) - 1)
    else:  # unrecognized overflow (完全静默处理)
        return - (pow(2, bit_width - 1) - 1)


def two_comp_to_true_(two_comp, bit_width=8, pad_zero=3):
    def two_comp_lit_to_ori(lit, _bit_width):  
        return - 1 * (2 ** (_bit_width - 1) - lit)

    if two_comp < 0:
        raise Exception("Error: not expecting negtive value")
    
    sign = two_comp >> (bit_width - 1)
    literal = two_comp & (2 ** (bit_width - 1) - 1)

    if sign == 0:  # positive value
        return literal
    elif sign == 4:  # positive value, 0100
        return literal
    elif sign == 1:  # positive overflow, 0001
        return pow(2, bit_width - 1) - 1
    elif sign == 3:  # negtive value, 0011
        return two_comp_lit_to_ori(literal, bit_width)
    elif sign == 7:  # negtive value, 0111
        return two_comp_lit_to_ori(literal, bit_width)
    elif sign == 6:  # negtive overflow, 0110
        return - (pow(2, bit_width - 1) - 1)
    else:  # unrecognized overflow (完全静默处理)
        return - (pow(2, bit_width - 1) - 1)


def restore_shape(component, shape, batch_size=16, bit_width=8, pad_zero=3):
    num_ele = np.prod(shape)
    num_ele_w_pad = batch_size * len(component)

    un_batched_nums = np.zeros(num_ele_w_pad, dtype=int)

    for i in range(batch_size):
        filter_ = (pow(2, bit_width + pad_zero) - 1) << ((bit_width + pad_zero) * i)
        for j in range(len(component)):
            two_comp = (filter_ & component[j]) >> ((bit_width + pad_zero) * i)
            un_batched_nums[batch_size * j + batch_size - 1 - i] = two_comp_to_true_(two_comp, bit_width, pad_zero)

    un_batched_nums = un_batched_nums[:num_ele]
    re = np.reshape(un_batched_nums, shape)
    return re


def decrypt_matrix_batch(private_key: PaillierPrivateKey, A, og_shape, batch_size=16, bit_width=8, pad_zero=3, r_max=0.5):
    decrypt_A = Parallel(n_jobs=N_JOBS)(delayed(private_key.decrypt)(num) for num in A)
    decrypt_A = np.array(decrypt_A)
    result = restore_shape(decrypt_A, og_shape, batch_size, bit_width, pad_zero)
    result = unquantize_matrix(result, bit_width, r_max)
    return result


def calculate_clip_threshold(grads, theta=2.5):
    return [theta * np.std(x) for x in grads]


def calculate_clip_threshold_sparse(grads, theta=2.5):
    result = []
    for layer in grads:
        if isinstance(layer, tf.IndexedSlices):
            result.append(theta * np.std(layer.values.numpy()))
        else:
            result.append(theta * np.std(layer.numpy()))
    return result


def clip_with_threshold(grads, thresholds):
    return [np.clip(x, -1 * y, y) for x, y in zip(grads, thresholds)]


def clip_gradients_std(grads, std_theta=2.5):
    results = []
    thresholds = []
    for component in grads:
        clip_T = np.std(component) * std_theta
        thresholds.append(clip_T)
        results.append(np.clip(component, -1 * clip_T, clip_T))
    return results, thresholds


def calculate_clip_threshold_aciq_g(grads, grads_sizes, bit_width=8):
    res = []
    for idx in range(len(grads)):
        res.append(aciq.get_alpha_gaus(grads[idx], grads_sizes[idx], bit_width))
    return res


def calculate_clip_threshold_aciq_l(grads, bit_width=8):
    return [aciq.get_alpha_laplace(x, bit_width) for x in grads]