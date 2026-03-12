# import os
# import tensorflow_datasets as tfds

# # 1. 强制禁用 GCS 访问
# os.environ['NO_GCE_CHECK'] = 'true'
# from tensorflow_datasets.core.utils import gcs_utils
# gcs_utils._is_gcs_disabled = True

# # 2. 如果你有代理，确保在代码中生效
# # os.environ['http_proxy'] = 'http://127.0.0.1:你的端口'
# # os.environ['https_proxy'] = 'http://127.0.0.1:你的端口'

# # 3. 再次尝试加载
# dataset, info = tfds.load(
#     'cifar10',
#     split='train',
#     shuffle_files=True,
#     with_info=True,
#     download=True
# )

#检查tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' # 显示所有详细日志，帮我们排错
import tensorflow as tf

print("\n" + "="*30)
print(f"TensorFlow 版本: {tf.__version__}")
print(f"Keras 版本: {tf.keras.__version__}")

# 强制触发一次 GPU 初始化
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"【成功】检测到 {len(gpus)} 个 GPU!")
        for gpu in gpus:
            print(f" - 设备详情: {gpu}")
        
        # 简单测试运算
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0]])
            b = tf.constant([[3.0], [4.0]])
            c = tf.matmul(a, b)
            print("【验证】GPU 矩阵运算成功！")
    else:
        print("【失败】依然未检测到 GPU。")
except Exception as e:
    print(f"【报错】初始化过程中出现异常: {e}")
print("="*30 + "\n")