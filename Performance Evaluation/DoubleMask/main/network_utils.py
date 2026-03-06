import socket
import struct
import pickle

def send_msg(sock, msg_dict):
    """
    发送字典消息到指定的 socket。
    采用 Header (4 bytes) + Payload 的协议格式解决 TCP 粘包问题。
    
    :param sock: socket 对象
    :param msg_dict: 需要发送的 Python 字典 (可包含 Numpy 数组等)
    """
    # 1. 序列化字典为字节流
    payload = pickle.dumps(msg_dict)
    
    # 2. 计算 payload 长度，并打包成 4 字节的 header 
    # '!I' 表示网络字节序 (大端序) 的 32 位无符号整型，上限支持 4GB 的单次传输
    header = struct.pack('!I', len(payload))
    
    # 3. 使用 sendall 确保底层完整发送 header + payload
    sock.sendall(header + payload)


def recvall(sock, n):
    """
    从 socket 接收严格大小为 n 的字节流。
    这是在网络受限环境下确保大文件传输不被截断的关键函数。
    
    :param sock: socket 对象
    :param n: 需要接收的字节总数
    :return: 接收到的 bytearray，如果对端正常关闭连接或网络中断则返回 None
    """
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


def recv_msg(sock):
    """
    从指定的 socket 接收并解析字典消息。
    
    :param sock: socket 对象
    :return: 解析后的 Python 字典，如果连接断开或异常返回 None
    """
    # 1. 先接收 4 字节的 header，获知接下来的数据体有多大
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    
    # 2. 解包获取 payload 的实际字节长度
    msglen = struct.unpack('!I', raw_msglen)[0]
    
    # 3. 根据长度，严格接收完整长度的数据体
    raw_payload = recvall(sock, msglen)
    if not raw_payload:
        return None
    
    # 4. 将完整的数据体反序列化为 Python 字典并返回
    return pickle.loads(raw_payload)