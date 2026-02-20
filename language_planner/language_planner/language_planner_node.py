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
        self.pose_sub = self.create_subscription(Odometry, '/mavros/vision_pose/pose', self.handle_pose, 1, callback_group=self.callback_group)
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
        self.map_pcl: np.ndarray = None
        self.freespace_pcl: np.ndarray = None
        self.target_waypoints = []
        self.target_ids = []


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

            
        # 修改 generate_plan 的调用，传入最新的层级信息
        self.target_waypoints, self.target_ids, filtered_objects_out, output_code = self.language_planner_backend.generate_plan(
            self.environment_name,
            input_statement,
            self.map_pcl,
            self.freespace_pcl,
            object_dict,
            self.cur_pos,
            scene_hierarchy=self.latest_hierarchy 
        )
        self.log_info(output_code)
        self.log_llm_output(input_statement, output_code)

        self.publish_object_markers(object_dict, self.target_ids)
        self.publish_waypoints(self.target_ids, self.target_waypoints, object_dict)

        
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