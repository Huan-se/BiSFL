import math

d = 2**20
N = 2**14
RU = 2**32
R = N * (RU-1) + 1
L1 = math.log2(RU)
L2 = math.log2(R)
factor = (9*1024*8+8*N*2+(d+N)*math.ceil(L2))/(d * math.ceil(L1))
print(factor)