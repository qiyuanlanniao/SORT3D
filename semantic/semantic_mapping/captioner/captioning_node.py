import os
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import argparse
import cv2
import pandas as pd
import open3d as o3d
import json
import torch
import torchvision.transforms.v2 as tt
from time import time
from enum import Enum
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path

from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String

from sensor_msgs_py.point_cloud2 import read_points_list

# from captioner.captioning_backend import Captioner
# from captioner.tools import ros2_bag_utils
from .captioning_backend import Captioner
from .tools import ros2_bag_utils
from std_srvs.srv import Trigger

class CaptioningNode(Node):

    def __init__(
            self,
            num_crop_levels: int = 3,
            crop_type = 'relative',
            crop_radius = 0.5,
            model_type = 'clip',
            process_terrain_map_into_freespace = False,
            simulator = 'wheelchair_unity',
            batch_size = 16
            ):

        # Variable Initialization

        self.cur_pos = np.array([0., 0., 0.])
        self.cur_orient = np.array([0., 0., 0., 0.])
        self.cur_rgb: torch.Tensor = None
        self.cur_sem: torch.Tensor = None
        self.queried_captions = None
        self.freespace_pcl: np.ndarray = None

        self.cur_pos_for_freespace: np.ndarray = None
        self.pos_change_threshold = 0.05

        self.batch_size = batch_size

        # Ground Truth Map Initialization

        self.sim_path = Path(__file__).resolve().parents[4] / "simulator" / simulator / "src" / "vehicle_simulator"
        self.scene_path = self.sim_path / 'mesh' / 'unity'

        self.device = "cuda:0"

        self.asset_list = pd.read_csv(str(self.scene_path / 'environment' /'AssetList.csv'))
        self.categories = pd.read_csv(str(self.scene_path / 'environment' / 'Categories.csv'))
        self.dimensions = pd.read_csv(str(self.scene_path / 'environment' / 'Dimensions.csv'))
        self.asset_list = self.asset_list[self.asset_list['name'].isin(self.categories['name'])].reset_index()


        self.semantic_ids = self.asset_list[['r', 'g', 'b']].to_numpy(dtype=np.uint8, copy=True)
        self.semantic_ids = np.hstack([self.semantic_ids, np.zeros((self.semantic_ids.shape[0], 1), dtype=np.uint8)])
        self.semantic_ids = np.ascontiguousarray(self.semantic_ids).view(np.uint32).flatten()

        semantic_dict = {}
        for i, id in enumerate(self.semantic_ids):
            
            centroid: np.ndarray = self.asset_list.loc[i, ['px', 'py', 'pz']].to_numpy()

            bounds_raw = self.dimensions.loc[
                self.dimensions['name'] == self.asset_list.loc[i, 'name'],
                ['min_x', 'max_x', 'min_y', 'max_y', 'min_z', 'max_z']
            ].squeeze().tolist()

            # 尝试安全转换为 float 数组
            try:
                bounds = np.array([float(x) for x in bounds_raw], dtype=np.float32)
            except ValueError:
                # self.get_logger().warning(f"Skipping object due to invalid bounds: {bounds_raw}")
                continue  # 跳过这个对象

            # 计算 dimensions
            dimensions: np.ndarray = bounds[[1, 3, 5]] - bounds[[0, 2, 4]]

            centroid[2] += dimensions[2] / 2

            q = self.asset_list.loc[i, ['qx', 'qy', 'qz', 'qw']].to_numpy()
            rpy = R.from_quat(q).as_euler('xyz')
            heading = rpy[2]

            l, w, h = dimensions.tolist()
            largest_face = max(
                l * w,
                w * h,
                l * h)

            semantic_dict[id] = {
                "image": {
                    "rgb": None,
                    "crop_coords": None,
                    "clip": None,
                    "caption": None,
                    "is_caption_generated": False
                    },
                "centroid": centroid.tolist(),
                "dimensions": dimensions.tolist(),
                "heading": heading,
                "largest_face": largest_face,
                "name": {
                    "string": self.categories[self.categories['name'] == self.asset_list.loc[i, 'name']]['cleaned'].iloc[0],
                    "similarity_to_image": None
                    }
                }
        #! 以上，直接透过仿真系统，得到了物体的真值
        # Captioner Backend

        self.image_shape = (640, 1920, 3)
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(f"dict size in captioningnode: {len(semantic_dict)}")
        self.captioner = Captioner(
            semantic_dict=semantic_dict,
            num_crop_levels=num_crop_levels,
            model_type=model_type,
            image_shape=self.image_shape,
            min_pixel_threshold=(28, 28),
            max_pixel_threshold=(640, 960), # should not occupy more than 120 deg horizontal fov
            crop_type=crop_type,
            crop_radius=crop_radius,
            device=self.device,
            log_info=self.log_info,
            load_captioner=True,
            crop_update_source="gt_semantics",
            batch_size=self.batch_size
        )

        # ROS

        super().__init__('captioning_node')

        ## Subs

        self.pose_sub = self.create_subscription(Odometry, '/state_estimation', self.handle_pose, 1, callback_group=MutuallyExclusiveCallbackGroup())
        self.rgb_sub = self.create_subscription(Image, '/camera/image', self.handle_rgb, 1, callback_group=MutuallyExclusiveCallbackGroup()) #! 回调，不做过多的处理
        self.sem_sub = self.create_subscription(Image, '/camera/semantic_image', self.handle_sem, 1, callback_group=MutuallyExclusiveCallbackGroup())
        self.query_sub = self.create_subscription(String, '/object_query', self.handle_object_query, 1, callback_group=MutuallyExclusiveCallbackGroup()) #! 来自于planner_node
        if process_terrain_map_into_freespace:
            self.freespace_sub = self.create_subscription(PointCloud2, '/terrain_map_ext', self.generate_freespace, 1, callback_group=MutuallyExclusiveCallbackGroup())

        ## Pubs

        self.caption_pub = self.create_publisher(String, '/queried_captions', 10)
        self.freespace_pub = self.create_publisher(PointCloud2, '/traversable_area', 5)# 可通行区域也是在这里完成的？

        ## Timers

        self.map_update_timer = self.create_timer(0.1, self.update_semantic_map, callback_group=MutuallyExclusiveCallbackGroup()) #! 周期性，根据最新的rgb和semantic图像，更新对应物体的最新crop
        self.caption_pub_timer = self.create_timer(0.1, self.publish_queried_captions, callback_group=MutuallyExclusiveCallbackGroup())#! 不断的发布最新的self.queried_captions，也就是带标题的物体列表

        # 缓存最新的全景图，用于验证
        self.latest_full_image = None
        
        # 创建服务
        self.rel_service = self.create_service(
            Trigger, # 实际建议自定义为 VerifyRelation.srv
            '/verify_spatial_relationship', 
            self.handle_verify_request
        )


    def log_info(self, message):
        self.get_logger().info(message)


    def handle_rgb(self, rgb: Image):
        self.cur_rgb = torch.\
            frombuffer(rgb.data, dtype=torch.uint8).\
            to(self.device).reshape(self.image_shape).flip((-1))
        
        self.latest_full_image = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')

    def handle_verify_request(self, request, response):
        if self.latest_full_image_cv is None:
            self.get_logger().error("❌ [VLM Service] 失败：当前没有可用的全景图像")
            response.success = False
            return response

        try:
            name_a, name_b = request.message.split(',')
            self.get_logger().info(f"📩 [VLM Service] 收到请求：验证 {name_a} 和 {name_b} 的关系")
            
            # 调用 VLM
            is_valid = self.captioner.verify_node_relationship(
                self.latest_full_image_cv, 
                name_a, 
                name_b
            )
            
            # 记录结果
            status = "YES (建立连接)" if is_valid else "NO (断开连接)"
            self.get_logger().info(f"🤖 [VLM Result] 模型判定结果: {status}")
            
            response.success = is_valid
            response.message = f"VLM marked as {status}"
            
        except Exception as e:
            self.get_logger().error(f"💥 [VLM Service] 运行崩溃: {str(e)}")
            response.success = True # 报错时默认不剪枝，防止误删
            
        return response
    
# --- topology_manager.py ---
from std_srvs.srv import Trigger

class TopologyManager(Node):
    def __init__(self):
        # ... 原有代码 ...
        # 创建服务客户端
        self.vlm_client = self.create_client(Trigger, '/verify_spatial_relationship')

    def call_vlm_verification_service(self, name_a, name_b):
        """
        同步调用 VLM 验证服务
        """
        if not self.vlm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("VLM Service not available, skipping pruning...")
            return True # 连不上服务时，默认保留边，防止误删

        req = Trigger.Request()
        req.message = f"{name_a},{name_b}" # 把两个物体名传过去

        # 发送同步请求 (在 Timer 中运行是安全的)
        future = self.vlm_client.call_async(req)
        
        # 简单的等待逻辑 (注意：真实环境建议用 async/await)
        import time
        start_time = time.time()
        while not future.done():
            if time.time() - start_time > 5.0: # 5秒超时
                return True
            time.sleep(0.1)

        return future.result().success


    def handle_sem(self, sem: Image):
        self.cur_sem = torch.frombuffer(sem.data, dtype=torch.uint8).to(self.device).reshape(self.image_shape).flip((-1))
        self.cur_sem = torch.cat(
            [self.cur_sem, 
            torch.zeros((self.cur_sem.shape[0], self.cur_sem.shape[1], 1), device=self.device, dtype=torch.uint8)], 
            dim=-1).view(torch.uint32)
        if self.cur_sem.shape != torch.Size([self.image_shape[0], self.image_shape[1], 1]):
            raise ValueError(f"Recasting uint8 into uint32 failed. Obtained shape {self.cur_sem.shape}") #! 像素级别的实例分割mask，与semantic ids对应


    def handle_pose(self, pose: Odometry):
        self.cur_pos[0] = pose.pose.pose.position.x
        self.cur_pos[1] = pose.pose.pose.position.y
        self.cur_pos[2] = pose.pose.pose.position.z

        self.cur_orient[0] = pose.pose.pose.orientation.w
        self.cur_orient[1] = pose.pose.pose.orientation.x
        self.cur_orient[2] = pose.pose.pose.orientation.y
        self.cur_orient[3] = pose.pose.pose.orientation.z


    def generate_freespace(self, msg: PointCloud2):
        if self.cur_pos_for_freespace is None or np.linalg.norm(self.cur_pos[:2] - self.cur_pos_for_freespace[:2]) > self.pos_change_threshold:
            pcd = read_points_list(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
            points_numpy = np.array(pcd).astype(np.float32)
            points_numpy = points_numpy[points_numpy[:, 3] < 0.05, :3]

            if self.freespace_pcl is None:
                self.freespace_pcl = points_numpy
            else:
                self.freespace_pcl = np.vstack([self.freespace_pcl, points_numpy])
            
            merged_pcd = o3d.geometry.PointCloud()
            merged_pcd.points = o3d.utility.Vector3dVector(
                self.freespace_pcl
            )

            voxel_size = 0.05
            merged_pcd = merged_pcd.voxel_down_sample(voxel_size)

            self.freespace_pcl = np.asarray(merged_pcd.points)

            self.cur_pos_for_freespace = self.cur_pos.copy()


            self.publish_freespace(self.freespace_pcl)

    
    def update_semantic_map(self):
        if self.cur_sem is None or self.cur_rgb is None:
            self.log_info(f'No images yet.')
            return
        self.captioner.update_object_crops(
            self.cur_rgb, 
            self.cur_sem,
            )


    def handle_object_query(self, query_str: String):
        query_list = json.loads(query_str.data)
        self.queried_captions = None # To stop publishing the captions in the other thread
        self.queried_captions = self.captioner.query_clip_features(query_list, self.cur_pos, self.cur_orient) #! 根据文本，使用clip计算文本和图像之间的距离，从dictionary中挑选物体


    def publish_freespace(self, freespace: np.ndarray):
        seconds, nanoseconds = self.get_clock().now().seconds_nanoseconds()
        msg = ros2_bag_utils.create_point_cloud(freespace[:, :3], seconds, nanoseconds, frame_id="map")
        self.freespace_pub.publish(msg)


    def publish_queried_captions(self):
        if self.queried_captions is None:
            return
        queried_caption_str = json.dumps(self.queried_captions)
        self.caption_pub.publish(String(data=queried_caption_str))


def main():
    rclpy.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_crop_levels', type=int, default=3)
    parser.add_argument('--model_type', default='clip')
    parser.add_argument('--process_terrain_map_into_freespace', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--simulator', default='wheelchair_unity')
    parser.add_argument('--batch_size', type=int, default=16)
    args, other_args = parser.parse_known_args()

    captioning_node = CaptioningNode(**vars(args))

    executor = MultiThreadedExecutor()
    executor.add_node(captioning_node)

    executor.spin()

    rclpy.shutdown()

if __name__=="__main__":
    main()