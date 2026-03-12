import numpy as np
import os
import open3d as o3d
import json
import argparse
import sys
from enum import Enum
from pathlib import Path
import cv2  
import glob 

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.time import Time
from geometry_msgs.msg import Pose2D, PointStamped, PoseStamped,TransformStamped, Point
from nav_msgs.msg import Odometry, Path as NavPath
from std_msgs.msg import String, ColorRGBA
from sensor_msgs.msg import PointCloud2, PointField, Image as ROSImage
from sensor_msgs_py.point_cloud2 import read_points_list, read_points_numpy
from visualization_msgs.msg import MarkerArray, Marker
import time

from language_planner.prompts import get_prompt
from language_planner.llm_backend.llm_query_langchain import NavQueryRunMode, ObjectQueryType, LanguageModel, SystemMode
from language_planner.language_planner_backend import LanguagePlannerBackend

from scipy.spatial.transform import Rotation as R

from captioner.tools import ros2_bag_utils
from captioner.captioning_backend import CropUpdateSource

from scipy.spatial.transform import Rotation as R

from tf2_ros import TransformBroadcaster
from cv_bridge import CvBridge


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

        self.path_pub = self.create_publisher(NavPath, '/robot_path_history', 10)
        self.path_history = NavPath()
        self.path_history.header.frame_id = "map" # 轨迹参考坐标系
        self.is_simulating_movement = False
        self.sim_target_pos = None
        self.move_speed = 0.5  # 模拟行走速度 0.5m/s
        
        # 路径记录容器 (使用我们刚刚起的别名 NavPath)
        self.path_history = NavPath()
        self.path_history.header.frame_id = "map"
        
        # 模拟位姿发布器（Bag停了，我们自己发位姿，箭头才会动）
        self.sim_pose_pub = self.create_publisher(PoseStamped, '/mavros/vision_pose/pose', 10)
        
        # 定义模拟器定时器 (10Hz)
        self.sim_timer = self.create_timer(0.1, self.sim_move_execution, callback_group=self.callback_group)

        self.tf_broadcaster = TransformBroadcaster(self)

        self.clear_timer = self.create_timer(1.0, self.clear_path_on_startup)

        self.target_image_pub = self.create_publisher(ROSImage, '/target_object_crop', 10)
        self.bridge = CvBridge()
        self.crops_base_path = "/home/iot/hm/ros2_ws/src/semantic/semantic_mapping/captioner/crops"

    def show_target_in_rviz(self, target_id, object_dict):
        """
        在 RViz 中高亮显示目标物体：发布抠图照片 + 在头上顶个大绿箭头
        """
        if target_id not in object_dict: return
        
        info = object_dict[target_id]
        pos = info['centroid']
        
        # --- 1. 检索图片文件 ---
        try:
            # 找到最新的日期文件夹 (2026-02-28_...)
            date_folders = sorted(glob.glob(os.path.join(self.crops_base_path, "*")))
            if not date_folders: return
            latest_date_folder = date_folders[-1]
            
            # 在日期文件夹里找以 "{ID}_" 开头的子文件夹
            target_folders = glob.glob(os.path.join(latest_date_folder, f"{target_id}_*"))
            if target_folders:
                crop_path = os.path.join(target_folders[0], "crop.png")
                if os.path.exists(crop_path):
                    # 读取并发布图片
                    cv_img = cv2.imread(crop_path)
                    ros_img = self.bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
                    self.target_image_pub.publish(ros_img)
                    self.log_info(f"🖼️ The cropping for target {target_id} has been published: {crop_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to extract target image: {e}")

        # --- 2. 发布巨大的绿色向下箭头 ---
        now = self.get_clock().now().to_msg()
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = now
        marker.ns = "target_indicator"
        marker.id = 9999 # 使用固定 ID 确保只有一个箭头
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # 设置箭头位置：在物体中心点上方 1.5 米处开始
        start_point = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]) + 1.5)
        # 终点：物体中心点
        end_point = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]) + 0.3)
        
        marker.points = [start_point, end_point]
        
        # 箭头粗细
        marker.scale.x = 0.15 # 轴直径
        marker.scale.y = 0.3  # 箭头直径
        marker.scale.z = 0.4  # 箭头长度
        
        # 亮绿色
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        
        self.object_marker_pub.publish(marker)
        
    def clear_path_on_startup(self):
        """
        启动时向 RViz 发送空路径，清理上一轮留下的视觉残留
        """
        # 1. 构造一个完全空的 Path 消息
        empty_path = NavPath()
        empty_path.header.frame_id = "map"
        empty_path.header.stamp = self.get_clock().now().to_msg()
        empty_path.poses = [] # 明确清空列表
        
        # 2. 发布清空信号
        self.path_pub.publish(empty_path)
        
        # 3. 同时也清空内存中的历史记录
        self.path_history = empty_path
        
        # 4. 销毁这个定时器，确保它只在启动时运行一次
        self.clear_timer.cancel()

    def sim_move_execution(self):
        """
        虚拟执行器：驱动机器人位置更新，并同步发布 Pose、Path 和 TF
        """
        if not self.is_simulating_movement or self.sim_target_pos is None:
            return

        # 1. 计算位移与剩余距离
        direction = self.sim_target_pos - self.cur_pos
        dist = np.linalg.norm(direction)

        # 到达判定
        if dist < 0.2:
            self.is_simulating_movement = False
            self.get_logger().info("🏁 Target point reached")
            return

        # 2. 更新内存中的坐标和朝向
        step = self.move_speed * 0.1 # 因为定时器是 10Hz (0.1s)
        self.cur_pos += (direction / dist) * step
        
        # 简单模拟转向：让机器人始终面向目标点
        yaw = np.arctan2(direction[1], direction[0])
        q = R.from_euler('z', yaw).as_quat()
        self.cur_orient = np.array([q[0], q[1], q[2], q[3]])

        # 3. 构造位姿消息 (PoseStamped)
        now = self.get_clock().now().to_msg()
        pose_msg = PoseStamped()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = float(self.cur_pos[0])
        pose_msg.pose.position.y = float(self.cur_pos[1])
        pose_msg.pose.position.z = float(self.cur_pos[2])
        pose_msg.pose.orientation.x = float(self.cur_orient[0])
        pose_msg.pose.orientation.y = float(self.cur_orient[1])
        pose_msg.pose.orientation.z = float(self.cur_orient[2])
        pose_msg.pose.orientation.w = float(self.cur_orient[3])

        # 发布 Pose 话题
        self.sim_pose_pub.publish(pose_msg) 

        # 4. 【关键】广播 TF 变换 (让 RViz 里的机器人模型/箭头动起来)
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link' # 必须对应你 RViz 中机器人的基础坐标系
        t.transform.translation.x = pose_msg.pose.position.x
        t.transform.translation.y = pose_msg.pose.position.y
        t.transform.translation.z = pose_msg.pose.position.z
        t.transform.rotation = pose_msg.pose.orientation
        self.tf_broadcaster.sendTransform(t)

        # 5. 记录并发布路径 (Path)
        self.path_history.header.stamp = now
        self.path_history.poses.append(pose_msg)
        # 限制路径长度，防止内存溢出
        if len(self.path_history.poses) > 2000:
            self.path_history.poses.pop(0)
        self.path_pub.publish(self.path_history)
        
        # 6. 打印移动状态
        self.get_logger().info(f"🏃 Moving... Distance remaining: {dist:.2f}m", throttle_duration_sec=1.0)

    def record_path_history(self, current_pose_msg):
        """记录并发布行走路径"""
        self.path_history.header.stamp = current_pose_msg.header.stamp
        self.path_history.poses.append(current_pose_msg)
        self.path_pub.publish(self.path_history)

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
        """
        修正版：支持跨越已知地图边界的长程导航
        """
        if not len(waypoints) or self.freespace_pcl is None: return
        freespace_points = self.freespace_pcl[:, :2]

        for i, (id_sublist, waypoint) in enumerate(zip(ids, waypoints)):
            # --- 核心改进：检查目标是否在已知地图内 ---
            dist_to_map_edge = np.linalg.norm(freespace_points - waypoint, axis=-1).min()
            
            # 如果目标点距离最近的已知空地超过 2 米，说明在“地图外”
            if dist_to_map_edge > 2.0:
                # self.get_logger().warn(f"🌐 目标 {waypoint} 在已知地图外，开启【长程探索】模式...")
                # 直接使用 LLM 给出的原始绝对坐标，不再进行裁剪
                target_xy = waypoint[:2]
            else:
                # 如果在地图内，则进行精确避障点映射
                closest_idx = np.linalg.norm(freespace_points - waypoint, axis=-1).argmin()
                target_xy = freespace_points[closest_idx]

            # --- 驱动模拟器 ---
            self.sim_target_pos = np.array([target_xy[0], target_xy[1], self.cur_pos[2]])
            self.is_simulating_movement = True
            
            self.get_logger().info(f"🚀 Target coordinates: {waypoint.round(2)} | Walk to: {target_xy.round(2)}")

            while self.is_simulating_movement and rclpy.ok():
                self.publish_waypoint(target_xy)
                
                # 每步都要检查是否进入感知范围（触发重感知截断）
                for obj_id in id_sublist:
                    if obj_id in self.object_dict:
                        if self.object_dict[obj_id]['relative_dist'] < self.focal_dist_threshold:
                            self.get_logger().info(f"👀 Approaching the target {obj_id}, prepare for detailed reasoning.")
                            self.is_simulating_movement = False
                            break
                time.sleep(0.1)
    

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


    def handle_language_query(self, msg: String, is_retry=False):
        self.log_info(f'Query received: {msg.data}')
        if not is_retry:
            self.path_history.poses = [] # 清空内存中的点
            self.path_pub.publish(self.path_history) # 发布空路径让 RViz 擦除线条
            # 可选：如果你想让之前的绿色箭头也消失，可以调用 self.clear_object_markers()
            self.clear_object_markers()

        # 定义统一的地图存放目录
        maps_dir = os.path.join(os.path.expanduser("~"), "hm/ros2_ws/maps")

        # 1. 检查并加载【可通行区域地图】 (Freespace)
        if self.freespace_pcl is None:
            map_path = os.path.join(maps_dir, "freespace_map.ply")
            if os.path.exists(map_path):
                self.log_info(f"📂 Loading passable map: {map_path}")
                try:
                    pcd = o3d.io.read_point_cloud(map_path)
                    self.freespace_pcl = np.asarray(pcd.points)
                except Exception as e:
                    self.log_info(f"❌ Failed to load passable map: {e}")
            else:
                self.log_info(f"⚠️ The topic has no data and the file cannot be found in {map_path}.")
                return

        # 2. 检查并加载【障碍物地图】 (Obstacles)
        if self.map_pcl is None:
            obs_path = os.path.join(maps_dir, "map_pcl.ply") # 统一命名为 map_pcl.ply
            if os.path.exists(obs_path):
                self.log_info(f"📂 Loading obstacle map: {obs_path}")
                try:
                    pcd_obs = o3d.io.read_point_cloud(obs_path)
                    self.map_pcl = np.asarray(pcd_obs.points)
                except Exception as e:
                    self.log_info(f"❌ Failed to load obstacle map: {e}")
            else:
                self.log_info(f"⚠️ The topic has no data and the file cannot be found in {obs_path}.")
                # 提示：即使没有障碍物地图，有些规划器也能跑，但建议保留

        # 3. 检查并加载【DSG层级结构描述】 (Scene Hierarchy)
        if not self.latest_hierarchy: # 检查字符串是否为空
            hierarchy_path = os.path.join(maps_dir, "latest_scene_graph.txt")
            if os.path.exists(hierarchy_path):
                self.log_info(f"📂 Loading hierarchy file: {hierarchy_path}")
                try:
                    with open(hierarchy_path, "r") as f:
                        self.latest_hierarchy = f.read()
                except Exception as e:
                    self.log_info(f"❌ Failed to load hierarchical files: {e}")
            else:
                self.log_info(f"❌ Critical error: Hierarchy description file {hierarchy_path} not found. The LLM will lose its context!")
                return
            pose_path = os.path.join(maps_dir, "last_robot_pose.json")
            if os.path.exists(pose_path):
                with open(pose_path, "r") as f:
                    pose_data = json.load(f)
                    self.cur_pos = np.array(pose_data["position"])
                    self.cur_orient = np.array(pose_data["orientation"])
                self.log_info(f"📍 The robot's initial position has been restored: {self.cur_pos}")
        
        input_statement = msg.data

        self.clear_object_markers()

        obj_query_list = self.language_planner_backend.get_object_references(input_statement)
        time.sleep(1)

        self.log_info(f'Parsed objects: {obj_query_list}')

        self.object_dict = {}
        self.obj_query_response_echo = None
        focal_candidate_ids = [] 

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
                time.sleep(0.1)
            self.log_info("Response received! Proceeding to LLM planning...")
            
            # 1. 获取包含“所有物体”的原始字典 (String keys)
            raw_all_dict = self.object_dict 
            self.save_scene_objects_json(raw_all_dict) 

            # 2. 调用 LLM 过滤，获取候选目标 ID 列表
            focal_candidate_ids = self.language_planner_backend.get_retrieved_objects(obj_query_list, raw_all_dict)
            
            # 3. 【核心修复】：构建一个完整的、标准化的 object_dict，并立即注入空间信息
            standardized_dict = {}
            for k_str, v_data in raw_all_dict.items():
                if k_str == "-1": continue # 跳过机器人
                
                # 统一坐标键名
                if "position" in v_data:
                    v_data["centroid"] = v_data["position"]
                
                # --- 关键：在这里立即计算并注入 local_pos ---
                dist, l_pos = self.get_relative_spatial_info(v_data['centroid'])
                v_data['relative_dist'] = dist
                v_data['local_pos'] = l_pos # 确保这个键存在！
                
                standardized_dict[int(k_str)] = v_data
            
            object_dict = standardized_dict # 替换为全量标准化字典

        # --- 4. 建立 Room 映射 (基于全量 object_dict) ---
        obj_to_room_map = {}
        if self.latest_hierarchy:
            import re
            room_sections = re.split(r'\[Room\]:', self.latest_hierarchy)
            for section in room_sections[1:]:
                room_id = section.strip().split('\n')[0].split(' ')[0].strip()
                ids = re.findall(r'ID:(\d+)', section)
                for oid in ids:
                    obj_to_room_map[int(oid)] = room_id

        # 5. 注入房间属性并确定机器人位置
        current_room_id = None
        min_dist = float('inf')
        for oid, info in object_dict.items():
            info['parent_room_id'] = obj_to_room_map.get(oid)
            # 确定机器人当前房间
            if info['relative_dist'] < min_dist and info['parent_room_id']:
                min_dist = info['relative_dist']
                current_room_id = info['parent_room_id']
        
        self.robot_current_room = current_room_id
        self.log_info(f"📍 Pre-calculation complete: The robot is currently located at {current_room_id}")

        # --- 6. 多分辨率裁剪 (利用前面存好的 local_pos) ---
        processed_dict_for_backend = {}
        peripheral_desc_lines = []
        room_summaries = {}

        for obj_id, info in object_dict.items():
            dist = info['relative_dist']
            rid = info.get('parent_room_id')
            
            # 判定 A: 是否在当前房间
            is_local_room = (rid == self.robot_current_room)
            # 判定 B: 是否在 3D 视野内 (前向象限)
            is_in_sight = self.is_in_sight_octants(info['local_pos'])
            # 判定 C: 是否在焦点距离内
            is_near = (dist < self.focal_dist_threshold)
            # 判定 D: 是否是 LLM 选出的候选目标 (来自 Filtered objects)
            is_candidate = (str(obj_id) in [str(x) for x in focal_candidate_ids])

            info_to_send = info.copy()

            if is_local_room:
                if is_candidate and is_near and is_in_sight:
                    # [TIER 1: 焦点区] - 保留所有信息 (Caption)
                    processed_dict_for_backend[obj_id] = info_to_send
                else:
                    # [TIER 2: 外围区] - 抹除 Caption 节省 Token
                    info_to_send['caption'] = "[Omitted]"
                    processed_dict_for_backend[obj_id] = info_to_send
                    peripheral_desc_lines.append(f"- {info['name']}(ID:{obj_id}, {dist:.1f}m)")
            else:
                # [TIER 3: 记忆区] - 汇总到房间概况，不进入 processed_dict
                if rid not in room_summaries:
                    room_summaries[rid] = []
                room_summaries[rid].append(info.get('name', 'item'))

        # 7. 生成全量 3D 罗盘观测 (用于让 LLM 看到远处的物体 ID)
        # 参数 1: TIER 1 罗盘描述 (仅显示当前房间的物体分布)
        local_room_objs = {k: v for k, v in processed_dict_for_backend.items() 
                           if v.get('parent_room_id') == self.robot_current_room}
        octant_desc = self.get_3d_octant_description(local_room_objs)

        # 参数 2: TIER 2 外围物体清单
        peripheral_desc = "=== Nearby Objects (Out of Sight or Far) ===\n"
        peripheral_desc += "\n".join(peripheral_desc_lines) if peripheral_desc_lines else "(None)"

        # 参数 3: TIER 3 全局记忆汇总
        memory_lines = []
        for rid, items in room_summaries.items():
            counts = {name: items.count(name) for name in set(items)}
            summary = ", ".join([f"{count} {name}(s)" for name, count in counts.items()])
            memory_lines.append(f"Room {rid} (Distant) contains: {summary}")
        global_memory_desc = "\n".join(memory_lines) if memory_lines else "(No other rooms detected)"

        # --- 8. 正式调用后端生成 Plan ---
        self.target_waypoints, self.target_ids, filtered_objects_out, output_code = self.language_planner_backend.generate_plan(
            self.environment_name,
            input_statement,
            self.map_pcl,
            self.freespace_pcl,
            processed_dict_for_backend, # 裁剪后的全量字典
            self.cur_pos,
            scene_hierarchy=self.latest_hierarchy,
            compass_description=octant_desc,        # 对应 TIER 1
            peripheral_description=peripheral_desc,   # 对应 TIER 2
            global_memory_description=global_memory_desc,
            full_object_dict=object_dict
        )

        # 触发目标视觉化
        if self.target_ids:
            # 假设取第一个识别出的 ID 作为主要目标
            primary_target_id = self.target_ids[0][0] 
            self.show_target_in_rviz(primary_target_id, object_dict)

        self.log_info(output_code)
        self.log_llm_output(input_statement, output_code)

        self.publish_object_markers(object_dict, self.target_ids)
        self.publish_waypoints(self.target_ids, self.target_waypoints, object_dict)
        target_reached_focal = False
        for sub_ids in self.target_ids:
            for oid in sub_ids:
                if oid in object_dict and object_dict[oid]['relative_dist'] < self.focal_dist_threshold:
                    target_reached_focal = True
        
        if not target_reached_focal and rclpy.ok():
            self.get_logger().info("🔄 [Re-perception]")
            # 延时一段时间，让机器狗多走一段距离
            time.sleep(2.0) 
            new_msg = String(data=input_statement)
            self.handle_language_query(new_msg, is_retry=True)
        else:
            self.get_logger().info("🏁 Mission accomplished: The target is now in sight and has been successfully reached.")

    def save_scene_objects_json(self, object_dict):
        """
        保存地图中“全量”物体信息，并包含机器狗（ID: -1）的实时位姿
        """
        maps_dir = os.path.join(os.path.expanduser("~"), "hm/ros2_ws/maps")
        os.makedirs(maps_dir, exist_ok=True)
        json_path = os.path.join(maps_dir, "scene_objects.json")

        all_objects_data = []
        
        # 1. 首先添加机器狗（Ego Robot）的信息
        # 利用类成员 self.cur_pos 获取当前最新的坐标
        all_objects_data.append({
            "id": -1,
            "name": "egorobot",
            "position": self.cur_pos.tolist(),
            "caption": "The ego robot being navigated around the room. It has an ID of -1. "
        })

        # 2. 遍历字典中的其他物体
        for obj_id_str, info in object_dict.items():
            # 跳过已经在上面手动添加过的 -1 节点
            if str(obj_id_str) == "-1":
                continue
                
            all_objects_data.append({
                "id": int(obj_id_str),
                "name": info.get("name", "unknown"),
                "position": info.get("centroid", [0.0, 0.0, 0.0]),
                "caption": info.get("caption", "No description available")
            })

        # 按 ID 排序（机器狗 -1 会排在最前面）
        all_objects_data.sort(key=lambda x: x["id"])

        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(all_objects_data, f, indent=4, ensure_ascii=False)
            self.log_info(f"✅ {len(all_objects_data)} objects, including the robot dog, have been saved to: {json_path}")
        except Exception as e:
            self.log_info(f"❌ Exporting JSON failed: {e}")

        
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