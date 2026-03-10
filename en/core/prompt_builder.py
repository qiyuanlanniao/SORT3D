# iot@iot:~/hm/ros2_ws/src/en/core/prompt_builder.py

import json
import numpy as np
from .geometry import OctantAnalyzer
from .extractor import TargetExtractor

class PromptBuilder:
    def __init__(self):
        self.analyzer = OctantAnalyzer()
        self.extractor = TargetExtractor()
        self.focal_distance = 3.0 # TIER 1 阈值
        
        self.system_context = (
            "You are an advanced robot spatial reasoner. Your goal is to identify "
            "the correct object ID. Respond ONLY with the command in the format: go_near(ID). "
            "Do not include the object name inside the parentheses. Example: go_near(5)"
        )

    def build_baseline_prompt(self, objects, instruction):
        """
        Baseline: 全量堆叠 (D1-D6 通用)
        """
        obj_list_str = []
        for obj in objects:
            if obj['id'] == -1: continue
            info = (
                f"- ID: {obj['id']}, Name: {obj['name']}, "
                f"Pos: {obj['position']}, Room: {obj['room']}, "
                f"Description: {obj['caption']}"
            )
            obj_list_str.append(info)

        return f"{self.system_context}\n\n=== Environment Objects ===\n" + \
               "\n".join(obj_list_str) + f"\n\n=== User Instruction ===\n{instruction}"

    def build_ours_prompt(self, robot_pos, objects, instruction):
        """
        Ours: 多分辨率空间注意力 (针对 D1-D6 优化)
        """
        # 1. 语义提取目标 (例如 "chair")
        target_category = self.extractor.extract(instruction)
        
        # 2. 确定机器人当前所在房间 (D1-D6 默认为 Room 0)
        # 实际开发中可以通过计算距离最近的节点或读取 -1 节点的 room 属性
        robot_room = 0 
        for obj in objects:
            if obj['id'] == -1:
                robot_room = obj.get('room', 0)
                break

        focal_tier = []      # TIER 1: 高细节 (近处或目标)
        peripheral_tier = []  # TIER 2: 低细节 (同房间远处背景)

        for obj in objects:
            if obj['id'] == -1: continue
            
            # 空间预计算
            spatial = self.analyzer.analyze(robot_pos, obj['position'])
            octant = spatial['octant']
            dist = spatial['distance']
            obj_room = obj.get('room', 0)

            # --- 分级过滤逻辑 ---
            is_target = target_category and target_category in obj['name'].lower()
            is_near = dist < self.focal_distance
            is_same_room = (obj_room == robot_room)

            if not is_same_room:
                # D1-D6 暂不处理跨房间逻辑，此处留空或简单记录
                continue

            # 判定是否进入 TIER 1 (焦点区)
            # 规则：3米内物体 OR 匹配指令的目标候选物
            if is_near or is_target:
                info = (
                    f"- {obj['name']}(ID: {obj['id']}): {octant}, {dist}m away. "
                    f"Detail: {obj['caption']}"
                )
                focal_tier.append(info)
            else:
                # 判定为 TIER 2 (外围区)：同房间的背景障碍物，抹除描述节省 Token
                info = f"- {obj['name']}(ID: {obj['id']}): {octant}, {dist}m"
                peripheral_tier.append(info)

        # 3. 组装 H-CoT 结构的 Prompt
        prompt = (
            f"{self.system_context}\n\n"
            f"### IMMEDIATE FOCAL SIGHT (Detailed descriptions) ###\n"
            + ("\n".join(focal_tier) if focal_tier else "None") + "\n\n"
            f"### PERIPHERAL AWARENESS (Landmarks without details) ###\n"
            + ("\n".join(peripheral_tier) if peripheral_tier else "None") + "\n\n"
            f"=== User Instruction ===\n"
            f"{instruction}\n\n"
            "Reasoning: 1. Search Focal Sight for the object matching the description. "
            "2. If not found, use Peripheral Awareness to find the correct direction. "
            "3. Output the command."
        )
        return prompt

# ---------------------------------------------------------
# 单元测试 (模拟 D6 场景)
# ---------------------------------------------------------
if __name__ == "__main__":
    builder = PromptBuilder()
    
    # 模拟 D6：机器人 [5,5,1]，目标在远处 [1.2, 0.8, 0.45] (约5.6米)
    mock_robot_pos = [5.0, 5.0, 1.0]
    mock_objects = [
        {"id": 1, "name": "chair", "position": [1.2, 0.8, 0.45], "room": 0, "caption": "A blue ergonomic chair."},
        {"id": 2, "name": "table", "position": [8.5, 1.5, 0.75], "room": 0, "caption": "A heavy green table."}
    ]
    # 如果指令找 chair，那么 chair 应该带 caption，table 不带
    mock_instruction = "Go to the blue ergonomic chair."

    print("--- Ours (D1-D6 Multi-res) Prompt Preview ---")
    print(builder.build_ours_prompt(mock_robot_pos, mock_objects, mock_instruction))