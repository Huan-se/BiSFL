import os
import re
import csv

def generate_summary():
    # 定义日志所在目录和输出文件
    results_dir = "benchmark_results"
    output_csv = "experiment_summary.csv"
    
    if not os.path.exists(results_dir):
        print(f"❌ 找不到结果目录 '{results_dir}'，请确保你当前在 Performance_Evaluation 目录下运行此脚本。")
        return
        
    csv_data = []
    # 定义将要导出的 CSV 表头
    headers = ["Scheme", "Clients", "Param_Power", "Param_Size", "Comm_Cost_KB", "Total_Time_s", "Status"]
    
    # 编译正则表达式，用于精准捕获日志中的浮点数指标
    re_comm = re.compile(r"总.*通信.*:\s*([0-9.]+)\s*KB")
    re_time = re.compile(r"总耗时.*:\s*([0-9.]+)\s*s")
    re_status = re.compile(r"\[(SUCCESS|TIMEOUT|FAILED|DIR_NOT_FOUND)\]")
    
    # 遍历 benchmark_results 目录下的所有文件
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith(".log") and "_clients_" in file:
                # 从文件名反向解析出方案名称和客户端数量
                parts = file.replace(".log", "").split("_clients_")
                if len(parts) != 2: 
                    continue
                
                scheme = parts[0]
                client_count = int(parts[1])
                
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # 利用我们 Bash 脚本中特有的配置打印行，将日志安全切分成多块
                blocks = content.split("测试配置 -> Params: ")
                
                # 第一个 block 是公用头部信息，我们从索引 1 开始遍历具体的参数实验块
                for block in blocks[1:]:
                    lines = block.strip().split('\n')
                    first_line = lines[0]
                    
                    # 提取参数的次方数和具体大小，如：2^10 (1024)
                    param_match = re.search(r"2\^(\d+)\s*\((\d+)\)", first_line)
                    if not param_match: 
                        continue
                        
                    p_pow = int(param_match.group(1))
                    p_size = int(param_match.group(2))
                    
                    # 初始化默认值
                    comm_val = None
                    time_val = None
                    status_val = "UNKNOWN"
                    
                    # 在当前实验的切块范围内检索关键指标
                    for line in lines:
                        m_comm = re_comm.search(line)
                        if m_comm: 
                            comm_val = float(m_comm.group(1))
                            
                        m_time = re_time.search(line)
                        if m_time: 
                            time_val = float(m_time.group(1))
                            
                        m_status = re_status.search(line)
                        if m_status: 
                            status_val = m_status.group(1)
                            
                    # 组装一条完整的数据记录
                    csv_data.append({
                        "Scheme": scheme,
                        "Clients": client_count,
                        "Param_Power": p_pow,
                        "Param_Size": p_size,
                        "Comm_Cost_KB": comm_val if comm_val is not None else "N/A",
                        "Total_Time_s": time_val if time_val is not None else "N/A",
                        "Status": status_val
                    })
                    
    if not csv_data:
        print("⚠️ 未能在日志中提取到任何有效结果，请检查日志内容。")
        return
        
    # 为了方便查阅，按 方案 -> 客户端规模 -> 参数维度 进行三重排序
    csv_data = sorted(csv_data, key=lambda x: (x["Scheme"], x["Clients"], x["Param_Power"]))
    
    # 写入 CSV 文件（使用 utf-8-sig 防止在 Windows Excel 下打开乱码）
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(csv_data)
        
    print(f"✅ 结果已成功提取并汇总至: {output_csv}\n")
    
    # 在终端直接打印出优雅的 Markdown 格式预览表格
    print(f"| {'Scheme':<12} | {'Clients':<8} | {'Param':<10} | {'Comm(KB)':<12} | {'Time(s)':<12} | {'Status':<10} |")
    print("|" + "-"*14 + "|" + "-"*10 + "|" + "-"*12 + "|" + "-"*14 + "|" + "-"*14 + "|" + "-"*12 + "|")
    for row in csv_data:
        comm_str = f"{row['Comm_Cost_KB']:.2f}" if isinstance(row['Comm_Cost_KB'], float) else str(row['Comm_Cost_KB'])
        time_str = f"{row['Total_Time_s']:.4f}" if isinstance(row['Total_Time_s'], float) else str(row['Total_Time_s'])
        param_str = f"2^{row['Param_Power']}"
        print(f"| {row['Scheme']:<12} | {row['Clients']:<8} | {param_str:<10} | {comm_str:<12} | {time_str:<12} | {row['Status']:<10} |")

if __name__ == "__main__":
    generate_summary()