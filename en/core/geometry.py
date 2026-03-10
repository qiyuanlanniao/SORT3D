# iot@iot:~/hm/ros2_ws/src/en/core/geometry.py

import numpy as np

class OctantAnalyzer:
    """
    立体 8 邻域（二阶魔方）分析器
    将全局坐标转换为以智能体为中心的相对方位和距离
    """
    def __init__(self, camera_height_offset=0.0):
        # 垂直分界偏移量，如果物体位置是相对于地面的，
        # 而机器人坐标是相对于相机中心的，可在此处调整
        self.h_cam = camera_height_offset

    def analyze(self, robot_pos, obj_pos):
        """
        输入:
            robot_pos: [x, y, z] 机器人的全局坐标
            obj_pos:   [x, y, z] 物体的全局坐标
        返回:
            dict: 包含相对距离、相对向量和 3D 象限标签
        """
        r = np.array(robot_pos)
        o = np.array(obj_pos)

        # 1. 计算 3D 相对位移向量
        # 假设坐标系定义：X为前后, Y为左右, Z为上下
        rel_vec = o - r

        # 2. 计算 3D 欧氏距离
        dist = np.linalg.norm(rel_vec)

        # 3. 判定立体 8 邻域 (3D Octant Binning)
        # 逻辑：根据相对位移的正负号决定它属于“二阶魔方”的哪个格子
        
        # 前后判定 (X轴)
        dx = "FRONT" if rel_vec[0] >= 0 else "BACK"
        
        # 左右判定 (Y轴)
        dy = "LEFT" if rel_vec[1] >= 0 else "RIGHT"
        
        # 上下判定 (Z轴)
        # 这里的判定基于物体中心相对于机器人高度的差值
        dz = "TOP" if rel_vec[2] >= self.h_cam else "BOTTOM"

        octant_label = f"{dx}_{dy}_{dz}"

        return {
            "distance": round(dist, 2),
            "octant": octant_label,
            "relative_vector": rel_vec.tolist()
        }

# ---------------------------------------------------------
# 单元测试 (测试 D3 数据集中的第一个物体)
# ---------------------------------------------------------
if __name__ == "__main__":
    analyzer = OctantAnalyzer()
    
    # D3 数据：Robot at [5, 5, 1], Chair at [6.2, 7.1, 1.8]
    test_robot = [5.0, 5.0, 1.0]
    test_chair = [6.2, 7.1, 1.8]
    
    result = analyzer.analyze(test_robot, test_chair)
    
    print("--- Geometry Test (D3 Unit) ---")
    print(f"Robot Pos: {test_robot}")
    print(f"Object Pos: {test_chair}")
    print(f"Distance: {result['distance']}m")
    print(f"Octant Sector: {result['octant']}")
    # 预期输出: FRONT_LEFT_TOP (因为 6.2>5, 7.1>5, 1.8>1)
