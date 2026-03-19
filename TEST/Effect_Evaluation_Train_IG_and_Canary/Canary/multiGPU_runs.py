# import os, sys, glob
# from tqdm import tqdm
# import myCMD

# EX = "python canary_attack_main.py %s %s"

# OUTP = './results/'

# try:
#     CONF = sys.argv[1]
#     num_run = int(sys.argv[2])
#     nGPU = int(sys.argv[3])
#     shift = int(sys.argv[4])
#     id_shift = int(sys.argv[5])
# except:
#     print("USAGE: 'CONF  NUM_RUNS nGPUs GPUidShift EXP_id_shift")
#     sys.exit(1)

# X = []
# for i in range(num_run):
#     ex = (CONF, i+id_shift)
#     X.append(ex)
# print(*X, sep='\n')

# myCMD.runMultiGPU(EX, X, nGPU, shift)



import os, sys
import myCMD

EX = "python canary_attack_main.py %s %s"
OUTP = './results/'

def main():
    # try:
    #     CONF = sys.argv[1]       # 实验配置文件
    #     num_run = int(sys.argv[2])  # 实验运行次数
    #     nGPU = int(sys.argv[3])     # 使用的 GPU 数量
    #     shift = int(sys.argv[4])    # GPU ID 偏移量
    #     id_shift = int(sys.argv[5]) # 实验 ID 偏移量
    # except:
    #     print("USAGE: python multiGPU_runs.py CONF NUM_RUNS nGPUs GPUidShift EXP_id_shift")
    #     sys.exit(1)
    CONF ="settings.c100_c10"
    num_run=300
    nGPU=9
    shift=0
    id_shift=450
    # 生成任务列表
    X = []
    for i in range(num_run):
        ex = (CONF, i + id_shift)
        X.append(ex)

    print("[INFO] 实验任务：")
    print(*X, sep='\n')

    # 运行多 GPU 任务
    myCMD.runMultiGPU(EX, X, nGPU, shift)


if __name__ == "__main__":
    # Windows 需要这个保护
    import multiprocessing as mp
    mp.freeze_support()  # 避免打包时的问题
    main()
