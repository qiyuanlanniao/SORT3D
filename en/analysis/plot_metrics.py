# iot@iot:~/hm/ros2_ws/src/en/analysis/plot_metrics.py

import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_results(dataset_name):
    # 获取当前脚本所在的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 动态构建结果文件夹路径 (en/results)
    results_dir = os.path.join(current_dir, "..", "results")
    
    base_file = os.path.join(results_dir, f"{dataset_name}_baseline_results.csv")
    ours_file = os.path.join(results_dir, f"{dataset_name}_ours_results.csv")
    
    if not os.path.exists(base_file) or not os.path.exists(ours_file):
        print(f"❌ Error: 找不到结果文件!")
        print(f"期待路径: \n1. {os.path.abspath(base_file)}\n2. {os.path.abspath(ours_file)}")
        return

    df_base = pd.read_csv(base_file)
    df_ours = pd.read_csv(ours_file)

    # 2. 计算核心统计指标
    stats = {
        "Avg Input Tokens (Base)": df_base["input_tokens"].mean(),
        "Avg Input Tokens (Ours)": df_ours["input_tokens"].mean(),
        "Avg Latency (Base)": df_base["latency"].mean(),
        "Avg Latency (Ours)": df_ours["latency"].mean(),
        "Success Rate (Base)": df_base["success"].mean() * 100,
        "Success Rate (Ours)": df_ours["success"].mean() * 100,
    }

    # 计算节省率
    reduction_rate = (stats["Avg Input Tokens (Base)"] - stats["Avg Input Tokens (Ours)"]) / stats["Avg Input Tokens (Base)"] * 100
    
    print(f"\n=== Experiment Summary for {dataset_name} ===")
    for k, v in stats.items():
        print(f"{k}: {v:.2f}")
    print(f"🔥 Token Reduction Rate: {reduction_rate:.2f}%")

    # 3. 绘制对比图表
    plt.style.use('seaborn-v0_8') # 使用学术风格
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- 图 1: Token 消耗对比 ---
    labels = ['Baseline', 'OmniVLN']
    tokens = [stats["Avg Input Tokens (Base)"], stats["Avg Input Tokens (Ours)"]]
    colors = ['#ff9999','#66b3ff']
    
    ax1.bar(labels, tokens, color=colors, width=0.6)
    ax1.set_title('Average Input Tokens per Query', fontsize=14)
    ax1.set_ylabel('Tokens')
    for i, v in enumerate(tokens):
        ax1.text(i, v + 5, f"{int(v)}", ha='center', fontweight='bold')

    # --- 图 2: 成功率对比 ---
    accuracy = [stats["Success Rate (Base)"], stats["Success Rate (Ours)"]]
    ax2.bar(labels, accuracy, color=colors, width=0.6)
    ax2.set_title('Navigation Decision Success Rate', fontsize=14)
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(0, 110)
    for i, v in enumerate(accuracy):
        ax2.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{dataset_name}_comparison.png"), dpi=300)

if __name__ == "__main__":
    analyze_results("D3")
