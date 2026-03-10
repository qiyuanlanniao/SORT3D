# iot@iot:~/hm/ros2_ws/src/en/scripts/run_baseline.py

import json
import csv
import os
import sys
import re

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.prompt_builder import PromptBuilder
from models.qwen_client import QwenVLMClient

def run_experiment():
    # 1. 初始化
    builder = PromptBuilder()
    client = QwenVLMClient()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "../data/D8.json")
    result_path = os.path.join(script_dir, "../results/D8_baseline_results.csv")
    
    with open(data_path, 'r') as f:
        all_data = json.load(f)

    # 提取物体列表（排除机器人）
    objects = [obj for obj in all_data if obj['id'] != -1]
    
    results = []

    # 2. 遍历每个有指令的物体进行测试
    
    for target_obj in objects:
        if 'instruction' not in target_obj: continue
        
        target_id = target_obj['id']
        instruction = target_obj['instruction']
        
        # 构造全量 Prompt
        prompt = builder.build_baseline_prompt(all_data, instruction)
        
        # 请求模型
        print(f"Testing ID {target_id}...")
        res = client.request(prompt)
        
        # 判定结果 (提取 go_near(ID) 里的数字)
        pred_id = -1
        if res["success"]:
            # 使用 findall 提取字符串中所有的数字，取最后一个
            ids_found = re.findall(r"\d+", res["answer"])
            if ids_found:
                pred_id = int(ids_found[-1])
        
        success = 1 if pred_id == target_id else 0
        
        # 记录数据
        results.append({
            "target_id": target_id,
            "input_tokens": res["input_tokens"],
            "output_tokens": res["output_tokens"],
            "latency": res["latency"],
            "success": success,
            "pred_id": pred_id,
            "answer": res["answer"].replace("\n", " ")
        })

    # 3. 保存结果
    keys = results[0].keys()
    os.makedirs("../results", exist_ok=True)
    with open(result_path, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
    
    print(f"✅ Baseline experiment finished. Saved to {result_path}")

if __name__ == "__main__":
    run_experiment()
