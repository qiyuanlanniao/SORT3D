# iot@iot:~/hm/ros2_ws/src/en/core/extractor.py

import re

class TargetExtractor:
    """
    语义关键字提取器 (修正版)
    逻辑：从指令中提取第一个出现的业务关键字作为主导航目标
    """
    def __init__(self):
        self.vocabulary = [
            "chair", 
            "table", 
            "screen", 
            "cabinet"
        ]

    def extract(self, instruction):
        """
        输入: "Find the small blue cabinet tucked under the table height."
        逻辑：找到所有匹配项，并返回在句子中位置最靠前的那个
        """
        text = instruction.lower()
        found_matches = []

        for category in self.vocabulary:
            # 使用 finditer 找到所有匹配的位置
            for match in re.finditer(rf'\b{category}\b', text):
                # 记录 (类别, 出现的位置索引)
                found_matches.append((category, match.start()))
        
        if not found_matches:
            return None
            
        # 按位置索引进行升序排序（最早出现的排在最前面）
        found_matches.sort(key=lambda x: x[1])

        # 返回第一个出现的关键词
        # 比如 "Find the cabinet... table..." -> cabinet 索引更小
        return found_matches[0][0]

# ---------------------------------------------------------
# 单元测试
# ---------------------------------------------------------
if __name__ == "__main__":
    extractor = TargetExtractor()
    
    test_cases = [
        "Navigate to the blue ergonomic office chair featuring a high backrest.",
        "Go to the green rectangular wooden table with a polished surface.",
        "Find the small blue cabinet tucked under the table height." # 关键测试项
    ]
    
    print("--- Corrected Extraction Test ---")
    for tc in test_cases:
        target = extractor.extract(tc)
        print(f"Instruction: {tc}")
        print(f"Extracted Target: {target}\n")
