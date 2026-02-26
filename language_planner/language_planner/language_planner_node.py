import numpy as np
import os
import open3d as o3d
import json
import argparse
import sys
from enum import Enum
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.time import Time
from geometry_msgs.msg import Pose2D, PointStamped, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py.point_cloud2 import read_points_list, read_points_numpy
from visualization_msgs.msg import MarkerArray, Marker
import time

from language_planner.prompts import get_prompt
from language_planner.llm_backend.llm_query_langchain import NavQueryRunMode, ObjectQueryType, LanguageModel, SystemMode
from language_planner.language_planner_backend import LanguagePlannerBackend

from captioner.tools import ros2_bag_utils
from captioner.captioning_backend import CropUpdateSource

from scipy.spatial.transform import Rotation as R


class PlatformType(Enum):
    WHEELCHAIR = 'wheelchair'
    MECANUM = 'mecanum'


class LanguagePlanner(Node):

    def __init__(
            self,
            environment_name: str,
            model='mistral',
            run_mode='use_tools',
            object_query_type='llm',
            platform: str = 'wheelchair'
            ):
        
        # Parameters

        self.environment_name = environment_name
        self.log_folder = Path(__file__).resolve().parents[1] / "logs" / "llm_outputs"
        if not os.path.exists(str(self.log_folder)):
            os.makedirs(self.log_folder)

        self.run_mode = NavQueryRunMode(run_mode)
        self.object_query_type = ObjectQueryType(object_query_type)
        self.model = LanguageModel(model)
        self.system_mode = SystemMode.LIVE_NAVIGATION
        self.platform = PlatformType(platform)
        # ROS

        super().__init__('language_planner_node')

        if self.platform == PlatformType.WHEELCHAIR:
            self.waypoint_pub = self.create_publisher(Pose2D, '/way_point_with_heading', 5)
        elif self.platform == PlatformType.MECANUM:
            self.waypoint_pub = self.create_publisher(PointStamped, '/goal_point', 5)
        self.object_query_pub = self.create_publisher(String, '/object_query', 5)
        self.object_marker_pub = self.create_publisher(Marker, '/selected_object_marker', 5)

        self.callback_group = ReentrantCallbackGroup()
        self.pose_sub = self.create_subscription(PoseStamped, '/mavros/vision_pose/pose', self.handle_pose, 1, callback_group=self.callback_group)
        self.map_sub = self.create_subscription(PointCloud2, '/cloud_registered', self.handle_map, 1, callback_group=self.callback_group)
        self.freespace_sub = self.create_subscription(PointCloud2, '/traversable_area', self.handle_freespace, 1, callback_group=self.callback_group)

        self.caption_sub = self.create_subscription(String, '/queried_captions', self.handle_captions, 1, callback_group=self.callback_group)
        self.planner_query_sub = self.create_subscription(String, '/language_planner_query', self.handle_language_query, 1, callback_group=self.callback_group)

        # 订阅来自 TopologyManager 的层级描述
        self.hierarchy_sub = self.create_subscription(
            String, 
            '/scene_hierarchy_description', 
            self.handle_hierarchy, 
            1, 
            callback_group=self.callback_group)

        self.latest_hierarchy = "" # 用于暂存层级信息

        self.cur_pos = np.array([0., 0., 0.])
        self.cur_vel = np.array([0., 0., 0.])
        self.cur_orient = np.array([0., 0., 0., 1.])
        self.map_pcl: np.ndarray = None
        self.freespace_pcl: np.ndarray = None
        self.target_waypoints = []
        self.target_ids = []
        self.focal_dist_threshold = 3.0      # 焦点区距离阈值（例如 4 米内的物体才看 Caption）
        self.camera_height_offset = 0.1     # 垂直分界线：相对于相机高度上下 0.5m 判定为 TOP/BOTTOM


        # Other

        self.language_planner_backend = LanguagePlannerBackend(
            self.model,
            self.run_mode,
            self.system_mode,
            self.log_info
            )

        self.object_dict = {}
        self.obj_query_response_echo = []

    def handle_hierarchy(self, msg: String):
        self.latest_hierarchy = msg.data
    
    def handle_pose(self, msg: PoseStamped):
        self.cur_pos[0] = msg.pose.position.x
        self.cur_pos[1] = msg.pose.position.y
        self.cur_pos[2] = msg.pose.position.z

        # --- 新增：记录朝向 ---
        self.cur_orient[0] = msg.pose.orientation.x
        self.cur_orient[1] = msg.pose.orientation.y
        self.cur_orient[2] = msg.pose.orientation.z
        self.cur_orient[3] = msg.pose.orientation.w

    def get_relative_spatial_info(self, target_pos):
        """
        第一阶段核心：3D 坐标变换。
        将全局坐标转换到以机器人为中心的局部坐标系 [前/后, 左/右, 上/下]
        """
        # 1. 计算全局 3D 相对位移
        rel_pos_global = np.array(target_pos) - self.cur_pos
        
        # 2. 获取机器人当前朝向的四元数逆旋转
        from scipy.spatial.transform import Rotation as R
        rot_robot = R.from_quat(self.cur_orient)
        
        # 3. 将位移旋转到机器人局部系：x'在前，y'在左，z'在上
        rel_pos_local = rot_robot.inv().apply(rel_pos_global)
        
        # 4. 计算 3D 欧氏距离
        distance = np.linalg.norm(rel_pos_local)
        
        return distance, rel_pos_local
    
    def get_3d_octant_description(self, object_dict):
        """
        第二阶段：基于局部坐标正负号，将空间划分为 8 个立体象限（类似二阶魔方）
        坐标系参考：x' 前/后, y' 左/右, z' 上/下
        """
        # 定义 8 个立体象限
        octants = {
            "FRONT_LEFT_TOP": [], "FRONT_RIGHT_TOP": [],
            "FRONT_LEFT_BOTTOM": [], "FRONT_RIGHT_BOTTOM": [],
            "BACK_LEFT_TOP": [], "BACK_RIGHT_TOP": [],
            "BACK_LEFT_BOTTOM": [], "BACK_RIGHT_BOTTOM": []
        }

        for obj_id, info in object_dict.items():
            if obj_id == -1: continue
            
            lp = info['local_pos'] # [x', y', z']
            dist = info['relative_dist']
            label = info.get('name', 'item')

            # 根据 x, y, z 的正负号决定象限
            dim_x = "FRONT" if lp[0] >= 0 else "BACK"
            dim_y = "LEFT" if lp[1] >= 0 else "RIGHT"
            dim_z = "TOP" if lp[2] >= self.camera_height_offset else "BOTTOM"
            
            key = f"{dim_x}_{dim_y}_{dim_z}"
            octants[key].append(f"{label}(ID:{obj_id}, {dist:.1f}m)")

        # 组装描述文本
        desc_lines = ["=== 3D Octant Observation (Agent-Centric) ==="]
        for key, items in octants.items():
            content = ", ".join(items) if items else "(Empty)"
            desc_lines.append(f"- {key}: {content}")
            
        return "\n".join(desc_lines)
    
    def is_in_sight_octants(self, local_pos):
        """
        判断物体是否在机器人前方的 4 个立体象限内（即“视野”内）
        逻辑：x' > 0 (前方)
        """
        # local_pos 格式为 [x', y', z']
        # 只要物体在机器人前方 (x' > 0)，我们就认为它是“焦点”
        return local_pos[0] > 0

    
    def handle_map(self, msg: PointCloud2):
        self.map_pcl = read_points_numpy(msg, ['x', 'y', 'z'], skip_nans=True)


    def handle_freespace(self, msg: PointCloud2):
        self.freespace_pcl = read_points_numpy(msg, ['x', 'y', 'z'], skip_nans=True)
   

    def handle_captions(self, msg: String):
        response_dict = json.loads(msg.data)
        self.object_dict = response_dict["response"]
        self.obj_query_response_echo = response_dict["query"]


    def publish_waypoint(self, waypoint: np.ndarray):
        if self.platform == PlatformType.WHEELCHAIR:
            waypoint_msg = Pose2D()
            waypoint_msg.x = float(waypoint[0]) + 0.0001
            waypoint_msg.y = float(waypoint[1]) + 0.0001
            waypoint_msg.theta = 0.
            self.waypoint_pub.publish(waypoint_msg)
        elif self.platform == PlatformType.MECANUM:
            waypoint_msg = PointStamped()
            waypoint_msg.header.frame_id = "map"
            waypoint_msg.point.x = float(waypoint[0]) + 0.0001
            waypoint_msg.point.y = float(waypoint[1]) + 0.0001
            waypoint_msg.point.z = 0.
            self.waypoint_pub.publish(waypoint_msg)
            self.log_info(f'PUBLISHED GOAL POINT')
    

    def publish_object_markers(self, object_dict, target_ids):

        seconds, nanoseconds = self.get_clock().now().seconds_nanoseconds()

        for id_sublist in target_ids:
            for i in id_sublist:
                bbox_marker = ros2_bag_utils.create_box_marker(
                    center=object_dict[i]["centroid"],
                    extent=object_dict[i]["dimensions"],
                    yaw=object_dict[i]["heading"],
                    ns="bboxes",
                    box_id=i,
                    color=[0., 0., 1., 0.6],
                    seconds=seconds,
                    nanoseconds=nanoseconds
                )

                text_height = 0.35
                text_position = [
                    object_dict[i]["centroid"][0],
                    object_dict[i]["centroid"][1],
                    object_dict[i]["centroid"][2] + object_dict[i]["dimensions"][2] + text_height/2 + 0.05,
                ]

                text_marker = ros2_bag_utils.create_text_marker(
                    center=text_position,
                    marker_id=i,
                    text=object_dict[i]["name"],
                    color=[1., 1., 1., 1.],
                    text_height=text_height,
                    seconds=seconds,
                    nanoseconds=nanoseconds
                )

                self.object_marker_pub.publish(bbox_marker)
                self.object_marker_pub.publish(text_marker)


    def recolor_object_markers(self, object_dict, target_ids):
        seconds, nanoseconds = self.get_clock().now().seconds_nanoseconds()

        for i in target_ids:
            bbox_marker = ros2_bag_utils.create_box_marker(
                center=object_dict[i]["centroid"],
                extent=object_dict[i]["dimensions"],
                yaw=object_dict[i]["heading"],
                ns="bboxes",
                box_id=i,
                color=[0., 1., 0., 0.6],
                seconds=seconds,
                nanoseconds=nanoseconds
            )

        self.object_marker_pub.publish(bbox_marker)


    def clear_object_markers(self):

        ns_list = ["text", "bboxes"]
        for id_sublist in self.target_ids:
            for idx in id_sublist:
                for ns in ns_list:
                    marker = Marker()
                    marker.header.frame_id = "map" 
                    seconds, nanoseconds = self.get_clock().now().seconds_nanoseconds()
                    marker.header.stamp = Time(seconds=seconds, nanoseconds=nanoseconds).to_msg()
                    marker.id = idx
                    marker.ns = ns
                    marker.type = Marker.CUBE
                    marker.action = Marker.DELETE
                    
                    self.object_marker_pub.publish(marker)

        self.target_ids = []
        self.target_waypoints = []


    def query_objects(self, object_nouns):
        object_json = json.dumps(object_nouns)
        object_msg = String(data=object_json)
        self.object_query_pub.publish(object_msg)


    def log_info(self, msg: str):
        self.get_logger().info(msg)


    def publish_waypoints(self, ids, waypoints, object_dict):
        if not len(waypoints):
            return

        freespace_points = self.freespace_pcl[:, :2]

        for i, (id_sublist, waypoint) in enumerate(zip(ids, waypoints)):
            closest_point_idx = np.linalg.norm(freespace_points - waypoint, axis=-1).argmin()
            closest_point = freespace_points[closest_point_idx]

            repeats = 5
            for k in range(repeats):
                self.publish_waypoint(closest_point)
                time.sleep(0.01)

            self.log_info(f"Navigating... ({i+1}/{len(waypoints)})")
            while rclpy.ok():
                dist_from_waypoint = np.linalg.norm(self.cur_pos[:2] - closest_point)
                if dist_from_waypoint < 1.5:
                    self.recolor_object_markers(object_dict, id_sublist)
                    break
        
        self.log_info(f'Finished navigation.')
    

    def log_llm_output(self, input_statement, output_code):
        seconds, nanoseconds = self.get_clock().now().seconds_nanoseconds()

        with open(str(self.log_folder / f'{seconds}.txt'), 'w') as f:
            f.write("===============\n")
            f.write("INPUT STATEMENT\n")
            f.write("===============\n")
            f.write(input_statement + '\n')
            f.write("===============\n")
            f.write("OUTPUT CODE\n")
            f.write("===============\n")
            f.write(output_code)


    def handle_language_query(self, msg: String):
        self.log_info(f'Query received: {msg.data}')

        # 定义统一的地图存放目录
        maps_dir = os.path.join(os.path.expanduser("~"), "hm/ros2_ws/maps")

        # 1. 检查并加载【可通行区域地图】 (Freespace)
        if self.freespace_pcl is None:
            map_path = os.path.join(maps_dir, "freespace_map.ply")
            if os.path.exists(map_path):
                self.log_info(f"📂 [Offline Mode] 正在加载可通行地图: {map_path}")
                try:
                    pcd = o3d.io.read_point_cloud(map_path)
                    self.freespace_pcl = np.asarray(pcd.points)
                except Exception as e:
                    self.log_info(f"❌ 加载可通行地图失败: {e}")
            else:
                self.log_info(f"⚠️ 话题无数据且在 {map_path} 找不到文件")
                return

        # 2. 检查并加载【障碍物地图】 (Obstacles)
        if self.map_pcl is None:
            obs_path = os.path.join(maps_dir, "map_pcl.ply") # 统一命名为 map_pcl.ply
            if os.path.exists(obs_path):
                self.log_info(f"📂 [Offline Mode] 正在加载障碍物地图: {obs_path}")
                try:
                    pcd_obs = o3d.io.read_point_cloud(obs_path)
                    self.map_pcl = np.asarray(pcd_obs.points)
                except Exception as e:
                    self.log_info(f"❌ 加载障碍物地图失败: {e}")
            else:
                self.log_info(f"⚠️ 话题无数据且在 {obs_path} 找不到文件")
                # 提示：即使没有障碍物地图，有些规划器也能跑，但建议保留

        # 3. 检查并加载【DSG层级结构描述】 (Scene Hierarchy)
        if not self.latest_hierarchy: # 检查字符串是否为空
            hierarchy_path = os.path.join(maps_dir, "latest_scene_graph.txt")
            if os.path.exists(hierarchy_path):
                self.log_info(f"📂 [Offline Mode] 正在加载层级结构文件: {hierarchy_path}")
                try:
                    with open(hierarchy_path, "r") as f:
                        self.latest_hierarchy = f.read()
                except Exception as e:
                    self.log_info(f"❌ 加载层级文件失败: {e}")
            else:
                self.log_info(f"❌ 严重错误：找不到层级描述文件 {hierarchy_path}，LLM 将失去上下文！")
                return
        
        input_statement = msg.data

        self.clear_object_markers()

        obj_query_list = self.language_planner_backend.get_object_references(input_statement)
        time.sleep(1)

        self.log_info(f'Parsed objects: {obj_query_list}')

        self.object_dict = {}
        self.obj_query_response_echo = None

        if self.object_query_type == ObjectQueryType.CLIP_BASED:
            self.query_objects(obj_query_list)
            self.log_info("Queried objects")
            while rclpy.ok() and self.obj_query_response_echo != obj_query_list:
                pass
            object_dict = self.object_dict
        elif self.object_query_type == ObjectQueryType.LLM_BASED:
            self.query_objects(["GET_ALL"])
            self.log_info("Queried objects")
            while rclpy.ok() and self.obj_query_response_echo != ["GET_ALL"]:
                # pass
                time.sleep(0.1)
            self.log_info("Response received! Proceeding to LLM planning...")
            
            # 这里的 self.object_dict 键是字符串 (来自 JSON)
            raw_object_dict = self.object_dict 
            self.save_scene_objects_json(raw_object_dict) 

            filtered_obj_ids = self.language_planner_backend.get_retrieved_objects(obj_query_list, raw_object_dict)
            
            if '-1' in raw_object_dict:
                filtered_obj_ids.append('-1')

            # 修复方案：强制转换 obj_id 为字符串去原字典查找，转换新字典键为整数
            new_object_dict = {}
            for obj_id in filtered_obj_ids:
                str_id = str(obj_id)
                if str_id in raw_object_dict:
                    new_object_dict[int(obj_id)] = raw_object_dict[str_id]
                else:
                    self.log_info(f"Warning: LLM suggested ID {obj_id}, but it's not in the scene!")
            
            object_dict = new_object_dict
            
        obj_to_room_map = {}
        if self.latest_hierarchy:
            import re
            room_sections = re.split(r'\[Room\]:', self.latest_hierarchy)
            for section in room_sections[1:]:
                room_id = section.strip().split('\n')[0].split(' ')[0].strip()
                ids = re.findall(r'ID:(\d+)', section)
                for oid in ids:
                    obj_to_room_map[int(oid)] = room_id

        # 注入属性
        for oid, info in object_dict.items():
            if oid != -1:
                info['parent_room_id'] = obj_to_room_map.get(int(oid))

        current_room_id = None
        min_dist = float('inf')
        for obj_id, info in object_dict.items():
            if obj_id == -1: continue
            dist, local_pos = self.get_relative_spatial_info(info['centroid'])
            info['relative_dist'] = dist
            info['local_pos'] = local_pos
            
            # 以离机器人最近的、且已知房间 ID 的物体来确定机器人位置
            if dist < min_dist and info.get('parent_room_id'):
                min_dist = dist
                current_room_id = info['parent_room_id']
        
        self.robot_current_room = current_room_id

        self.get_logger().info(f"📍 预计算完成：机器人当前位于 {current_room_id}, 已计算 {len(object_dict)} 个物体的 3D 相对位姿")

        focal_objects = {}      # 高分辨率：详细 Caption + 8 邻域方位
        peripheral_objects = {} # 中分辨率：仅 ID + 标签 + 距离
        room_summaries = {}     # 低分辨率：仅房间概况

        processed_dict_for_backend = {}

        for obj_id, info in object_dict.items():
            if obj_id == -1: continue
            
            dist = info['relative_dist']
            is_near = (dist < self.focal_dist_threshold)
            is_in_sight = self.is_in_sight_octants(info['local_pos'])
            
            # 拷贝一份数据，避免修改原始缓存
            info_to_send = info.copy()

            # --- 分辨率决策 ---
            if is_near and is_in_sight:
                # 【高分辨率】在眼前且近：保留所有，LLM 能看到详细 Caption
                pass 
            else:
                # 【低分辨率】在身后或太远：抹除 Caption，仅保留坐标和 ID
                # 这样 LLM 知道它在哪，但不会被长文本淹没，也不会浪费 Token
                info_to_send['caption'] = "[Description omitted due to distance/view]"
            
            processed_dict_for_backend[obj_id] = info_to_send

        # 2. 这里的 octant_desc 应该包含所有 processed_dict_for_backend 里的方位
        # 这样 LLM 才能看到 "BACK: chair(ID:68, 22.8m)"
        full_octant_desc = self.get_3d_octant_description(processed_dict_for_backend)

        # 生成房间概况文本（零 Token 预压缩）
        room_desc_lines = []
        for rid, items in room_summaries.items():
            counts = {name: items.count(name) for name in set(items)}
            summary = ", ".join([f"{count} {name}(s)" for name, count in counts.items()])
            room_desc_lines.append(f"Room {rid} (Far away) contains: {summary}")
        
        global_memory_desc = "\n".join(room_desc_lines)
        # 1. 组装外围物体的简要清单 (Mid-Res)
        peripheral_desc = "=== Nearby Objects (Out of Sight or Far) ===\n"
        for oid, info in peripheral_objects.items():
            peripheral_desc += f"- {info['name']}(ID:{oid}, {info['relative_dist']:.1f}m)\n"
        if not peripheral_objects: peripheral_desc += "(None)\n"

        local_room_objects = {oid: info for oid, info in object_dict.items() 
                              if info.get('parent_room_id') == current_room_id}
        
        # 这个描述会告诉 LLM：即便在 BACK（后方），那里也有个椅子
        octant_desc = self.get_3d_octant_description(local_room_objects)

        # 打印日志方便你调试
        self.get_logger().info(f"\n[DEBUG] 当前 3D 焦点观测：\n{octant_desc}")
            
        # --- 对比实验开关 ---
        USE_HIERARCHY = True  # 改为 False 即关闭层次化
        # ------------------

        if not USE_HIERARCHY:
            # 1. 清空层级描述，让 LLM 拿不到房间和组的信息
            display_hierarchy = "" 
            self.log_info("🧪 [Experiment] 正在以【非层次化/扁平模式】运行推理...")
        else:
            display_hierarchy = self.latest_hierarchy

        # 3. 修改后端调用：传入三级分辨率数据
        self.target_waypoints, self.target_ids, filtered_objects_out, output_code = self.language_planner_backend.generate_plan(
            self.environment_name,
            input_statement,
            self.map_pcl,
            self.freespace_pcl,
            processed_dict_for_backend,  # 只有这里的物体带有冗长的 Caption
            self.cur_pos,
            scene_hierarchy=display_hierarchy,
            compass_description=full_octant_desc, # 这里的 octant_desc 仅包含 focal_objects 的方位
            peripheral_description=peripheral_desc, # 新增
            global_memory_description=global_memory_desc # 新增
        )

        self.log_info(output_code)
        self.log_llm_output(input_statement, output_code)

        self.publish_object_markers(object_dict, self.target_ids)
        self.publish_waypoints(self.target_ids, self.target_waypoints, object_dict)

    def save_scene_objects_json(self, object_dict):
        """
        保存地图中“全量”识别物体的详情
        """
        maps_dir = os.path.join(os.path.expanduser("~"), "hm/ros2_ws/maps")
        os.makedirs(maps_dir, exist_ok=True)
        json_path = os.path.join(maps_dir, "scene_objects.json")

        all_objects_data = []
        
        # 遍历原始字典中的每一个物体
        for obj_id_str, info in object_dict.items():
            # 跳过机器人节点 (-1)
            if str(obj_id_str) == "-1":
                continue
                
            all_objects_data.append({
                "id": int(obj_id_str), # 确保 ID 是整数
                "name": info.get("name", "unknown"),
                "position": info.get("centroid", [0.0, 0.0, 0.0]),
                "caption": info.get("caption", "No description available")
            })

        # 按 ID 排序，方便查看
        all_objects_data.sort(key=lambda x: x["id"])

        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(all_objects_data, f, indent=4, ensure_ascii=False)
            self.log_info(f"✅ [Full Export] 全量物体信息({len(all_objects_data)}个)已保存至: {json_path}")
        except Exception as e:
            self.log_info(f"❌ 导出全量 JSON 失败: {e}")

        
def main():
    rclpy.init(args=sys.argv)

    parser = argparse.ArgumentParser()
    parser.add_argument('--environment_name', default='a room')
    parser.add_argument('--model', default='mistral')
    parser.add_argument('--run_mode', default='use_tools')
    parser.add_argument('--object_query_type', default='llm')
    parser.add_argument('--platform', default='wheelchair')
    argparse_args, other_args = parser.parse_known_args()

    handler = LanguagePlanner(**vars(argparse_args))

    executor = MultiThreadedExecutor()
    executor.add_node(handler)

    executor.spin()

    rclpy.shutdown()


if __name__ == "__main__":
    main()