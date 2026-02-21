import numpy as np
import scipy.ndimage
from scipy.spatial.transform import Rotation
import cv2
import scipy

def scan2pixels(laserCloud, L2C_PARA, CAMERA_PARA, LIDAR_PARA):
    lidarX = L2C_PARA["x"] #   lidarXStack[imageIDPointer]
    lidarY = L2C_PARA["y"] # idarYStack[imageIDPointer]
    lidarZ = L2C_PARA["z"] # lidarZStack[imageIDPointer]
    lidarRoll = -L2C_PARA["roll"] #  lidarRollStack[imageIDPointer]
    lidarPitch = -L2C_PARA["pitch"] # lidarPitchStack[imageIDPointer]
    lidarYaw = -L2C_PARA["yaw"]# lidarYawStack[imageIDPointer]

    imageWidth = CAMERA_PARA["width"]
    imageHeight = CAMERA_PARA["height"]
    cameraOffsetZ = 0   #  additional pixel offset due to image cropping? 
    vertPixelOffset = 0  #  additional vertical pixel offset due to image cropping

    sinLidarRoll = np.sin(lidarRoll)
    cosLidarRoll = np.cos(lidarRoll)
    sinLidarPitch = np.sin(lidarPitch)
    cosLidarPitch = np.cos(lidarPitch)
    sinLidarYaw = np.sin(lidarYaw)
    cosLidarYaw = np.cos(lidarYaw)
    
    lidar_offset = np.array([lidarX, lidarY, lidarZ])
    camera_offset = np.array([0, 0, cameraOffsetZ])
    
    cloud = laserCloud[:, :3] - lidar_offset
    R_z = np.array([[cosLidarYaw, -sinLidarYaw, 0], [sinLidarYaw, cosLidarYaw, 0], [0, 0, 1]])
    R_y = np.array([[cosLidarPitch, 0, sinLidarPitch], [0, 1, 0], [-sinLidarPitch, 0, cosLidarPitch]])
    R_x = np.array([[1, 0, 0], [0, cosLidarRoll, -sinLidarRoll], [0, sinLidarRoll, cosLidarRoll]])
    cloud = cloud @ R_z @ R_y @ R_x
    cloud = cloud - camera_offset
    
    horiDis = np.sqrt(cloud[:, 0] ** 2 + cloud[:, 1] ** 2)
    horiPixelID = (-imageWidth / (2 * np.pi) * np.arctan2(cloud[:, 1], cloud[:, 0]) + imageWidth / 2 + 1).astype(int) - 1
    vertPixelID = (-imageWidth / (2 * np.pi) * np.arctan2(cloud[:, 2], horiDis) + imageHeight / 2 + 1 + vertPixelOffset).astype(int)
    PixelDepth = horiDis

    horiPixelID = np.clip(horiPixelID, 0, CAMERA_PARA["width"] - 1)
    vertPixelID = np.clip(vertPixelID, 0, CAMERA_PARA["height"] - 1)

    point_pixel_idx = np.array([horiPixelID, vertPixelID, PixelDepth]).T
    
    return point_pixel_idx.astype(int)

def scan2pixels_wheelchair(laserCloud):
    # project scan points to image pixels
    # https://github.com/jizhang-cmu/cmu_vla_challenge_unity/blob/noetic/src/semantic_scan_generation/src/semanticScanGeneration.cpp
    
    # Input: 
    # [#points, 3], x-y-z coordinates of lidar points
    
    # Output: 
    #    point_pixel_idx['horiPixelID'] : horizontal pixel index in the image coordinate
    #    point_pixel_idx['vertPixelID'] : vertical pixel index in the image coordinate

    # L2C_PARA= {"x": 0, "y": 0, "z": 0.235, "roll": -1.5707963, "pitch": 0, "yaw": -1.5707963} #  mapping from scan coordinate to camera coordinate(m) (degree), camera is  "z" higher than lidar
    L2C_PARA= {"x": 0, "y": 0, "z": 0.235, "roll": 0.0, "pitch": 0, "yaw": -0.0} #  mapping from scan coordinate to camera coordinate(m) (degree), camera is  "z" higher than lidar
    CAMERA_PARA= {"hfov": 360, "vfov": 120, "width": 1920, "height": 640}  # cropped 30 degree(160 pixels) in top and  30 degree(160 pixels) in bottom 
    LIDAR_PARA= {"hfov": 360, "vfov": 30}   
    
    return scan2pixels(laserCloud, L2C_PARA, CAMERA_PARA, LIDAR_PARA)

def scan2pixels_mecanum_sim(laserCloud):
    CAMERA_PARA= {"x": 0.0, "y": 0.0, "z": 0.1, "roll": -1.5707963, "pitch": 0, "yaw": -1.5707963, "hfov": 360, "vfov": 120, "width": 1920, "height": 640}  # cropped 30 degree(160 pixels) in top and  30 degree(160 pixels) in bottom 
    LIDAR_PARA= {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}

    lidar_offset = np.array([LIDAR_PARA["x"], LIDAR_PARA["y"], LIDAR_PARA["z"]])
    lidarRoll = LIDAR_PARA["roll"] #  lidarRollStack[imageIDPointer]
    lidarPitch = LIDAR_PARA["pitch"] # lidarPitchStack[imageIDPointer]
    lidarYaw = LIDAR_PARA["yaw"]# lidarYawStack[imageIDPointer]
    lidarR_z = np.array([[np.cos(lidarYaw), -np.sin(lidarYaw), 0], [np.sin(lidarYaw), np.cos(lidarYaw), 0], [0, 0, 1]])
    lidarR_y = np.array([[np.cos(lidarPitch), 0, np.sin(lidarPitch)], [0, 1, 0], [-np.sin(lidarPitch), 0, np.cos(lidarPitch)]])
    lidarR_x = np.array([[1, 0, 0], [0, np.cos(lidarRoll), -np.sin(lidarRoll)], [0, np.sin(lidarRoll), np.cos(lidarRoll)]])
    lidarR = lidarR_z @ lidarR_y @ lidarR_x

    cam_offset = np.array([CAMERA_PARA["x"], CAMERA_PARA["y"], CAMERA_PARA["z"]])
    camRoll = CAMERA_PARA["roll"]
    camPitch = CAMERA_PARA["pitch"]
    camYaw = CAMERA_PARA["yaw"]
    camR_z = np.array([[np.cos(camYaw), -np.sin(camYaw), 0], [np.sin(camYaw), np.cos(camYaw), 0], [0, 0, 1]])
    camR_y = np.array([[np.cos(camPitch), 0, np.sin(camPitch)], [0, 1, 0], [-np.sin(camPitch), 0, np.cos(camPitch)]])
    camR_x = np.array([[1, 0, 0], [0, np.cos(camRoll), -np.sin(camRoll)], [0, np.sin(camRoll), np.cos(camRoll)]])
    camR = camR_z @ camR_y @ camR_x

    xyz = laserCloud[:, :3] - lidar_offset
    xyz = xyz @ lidarR
    xyz = xyz - cam_offset
    xyz = xyz @ camR

    horiDis = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 2] ** 2)
    horiPixelID = (CAMERA_PARA["width"] / (2 * np.pi) * np.arctan2(xyz[:, 0], xyz[:, 2]) + CAMERA_PARA["width"] / 2 + 1).astype(int)
    vertPixelID = (CAMERA_PARA["width"] / (2 * np.pi) * np.arctan(xyz[:, 1] / horiDis) + CAMERA_PARA["height"] / 2 + 1).astype(int)
    pixelDepth = horiDis

    horiPixelID = np.clip(horiPixelID, 0, CAMERA_PARA["width"] - 1)
    vertPixelID = np.clip(vertPixelID, 0, CAMERA_PARA["height"] - 1)

    point_pixel_idx = np.array([horiPixelID, vertPixelID, pixelDepth]).T
    
    return point_pixel_idx

def scan2pixels_mecanum(laserCloud):
    CAMERA_PARA= {"x": -0.12, "y": -0.075, "z": 0.265, "roll": -1.5707963, "pitch": 0, "yaw": -1.5707963, "hfov": 360, "vfov": 120, "width": 1920, "height": 640}  # cropped 30 degree(160 pixels) in top and  30 degree(160 pixels) in bottom 
    LIDAR_PARA= {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}

    lidar_offset = np.array([LIDAR_PARA["x"], LIDAR_PARA["y"], LIDAR_PARA["z"]])
    lidarRoll = LIDAR_PARA["roll"] #  lidarRollStack[imageIDPointer]
    lidarPitch = LIDAR_PARA["pitch"] # lidarPitchStack[imageIDPointer]
    lidarYaw = LIDAR_PARA["yaw"]# lidarYawStack[imageIDPointer]
    lidarR_z = np.array([[np.cos(lidarYaw), -np.sin(lidarYaw), 0], [np.sin(lidarYaw), np.cos(lidarYaw), 0], [0, 0, 1]])
    lidarR_y = np.array([[np.cos(lidarPitch), 0, np.sin(lidarPitch)], [0, 1, 0], [-np.sin(lidarPitch), 0, np.cos(lidarPitch)]])
    lidarR_x = np.array([[1, 0, 0], [0, np.cos(lidarRoll), -np.sin(lidarRoll)], [0, np.sin(lidarRoll), np.cos(lidarRoll)]])
    lidarR = lidarR_z @ lidarR_y @ lidarR_x

    cam_offset = np.array([CAMERA_PARA["x"], CAMERA_PARA["y"], CAMERA_PARA["z"]])
    camRoll = CAMERA_PARA["roll"]
    camPitch = CAMERA_PARA["pitch"]
    camYaw = CAMERA_PARA["yaw"]
    camR_z = np.array([[np.cos(camYaw), -np.sin(camYaw), 0], [np.sin(camYaw), np.cos(camYaw), 0], [0, 0, 1]])
    camR_y = np.array([[np.cos(camPitch), 0, np.sin(camPitch)], [0, 1, 0], [-np.sin(camPitch), 0, np.cos(camPitch)]])
    camR_x = np.array([[1, 0, 0], [0, np.cos(camRoll), -np.sin(camRoll)], [0, np.sin(camRoll), np.cos(camRoll)]])
    camR = camR_z @ camR_y @ camR_x

    xyz = laserCloud[:, :3] - lidar_offset
    xyz = xyz @ lidarR
    xyz = xyz - cam_offset
    xyz = xyz @ camR

    horiDis = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 2] ** 2)
    horiPixelID = (CAMERA_PARA["width"] / (2 * np.pi) * np.arctan2(xyz[:, 0], xyz[:, 2]) + CAMERA_PARA["width"] / 2 + 1).astype(int)
    vertPixelID = (CAMERA_PARA["width"] / (2 * np.pi) * np.arctan(xyz[:, 1] / horiDis) + CAMERA_PARA["height"] / 2 + 1).astype(int)
    pixelDepth = horiDis

    horiPixelID = np.clip(horiPixelID, 0, CAMERA_PARA["width"] - 1)
    vertPixelID = np.clip(vertPixelID, 0, CAMERA_PARA["height"] - 1)
    point_pixel_idx = np.array([horiPixelID, vertPixelID, pixelDepth]).T
    
    return point_pixel_idx

def scan2pixels_diablo(laserCloud):
    CAMERA_PARA= {"x": 0.0, "y": 0.0, "z": 0.185, "roll": -1.5707963, "pitch": 0, "yaw": -1.5707963, "hfov": 360, "vfov": 120, "width": 1920, "height": 640}  # cropped 30 degree(160 pixels) in top and  30 degree(160 pixels) in bottom 
    LIDAR_PARA= {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}

    lidar_offset = np.array([LIDAR_PARA["x"], LIDAR_PARA["y"], LIDAR_PARA["z"]])
    lidarRoll = LIDAR_PARA["roll"] #  lidarRollStack[imageIDPointer]
    lidarPitch = LIDAR_PARA["pitch"] # lidarPitchStack[imageIDPointer]
    lidarYaw = LIDAR_PARA["yaw"]# lidarYawStack[imageIDPointer]
    lidarR_z = np.array([[np.cos(lidarYaw), -np.sin(lidarYaw), 0], [np.sin(lidarYaw), np.cos(lidarYaw), 0], [0, 0, 1]])
    lidarR_y = np.array([[np.cos(lidarPitch), 0, np.sin(lidarPitch)], [0, 1, 0], [-np.sin(lidarPitch), 0, np.cos(lidarPitch)]])
    lidarR_x = np.array([[1, 0, 0], [0, np.cos(lidarRoll), -np.sin(lidarRoll)], [0, np.sin(lidarRoll), np.cos(lidarRoll)]])
    lidarR = lidarR_z @ lidarR_y @ lidarR_x

    cam_offset = np.array([CAMERA_PARA["x"], CAMERA_PARA["y"], CAMERA_PARA["z"]])
    camRoll = CAMERA_PARA["roll"]
    camPitch = CAMERA_PARA["pitch"]
    camYaw = CAMERA_PARA["yaw"]
    camR_z = np.array([[np.cos(camYaw), -np.sin(camYaw), 0], [np.sin(camYaw), np.cos(camYaw), 0], [0, 0, 1]])
    camR_y = np.array([[np.cos(camPitch), 0, np.sin(camPitch)], [0, 1, 0], [-np.sin(camPitch), 0, np.cos(camPitch)]])
    camR_x = np.array([[1, 0, 0], [0, np.cos(camRoll), -np.sin(camRoll)], [0, np.sin(camRoll), np.cos(camRoll)]])
    camR = camR_z @ camR_y @ camR_x

    xyz = laserCloud[:, :3] - lidar_offset
    xyz = xyz @ lidarR
    xyz = xyz - cam_offset
    xyz = xyz @ camR

    horiDis = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 2] ** 2)
    horiPixelID = (CAMERA_PARA["width"] / (2 * np.pi) * np.arctan2(xyz[:, 0], xyz[:, 2]) + CAMERA_PARA["width"] / 2 + 1).astype(int)
    vertPixelID = (CAMERA_PARA["width"] / (2 * np.pi) * np.arctan(xyz[:, 1] / horiDis) + CAMERA_PARA["height"] / 2 + 1).astype(int)
    pixelDepth = horiDis

    horiPixelID = np.clip(horiPixelID, 0, CAMERA_PARA["width"] - 1)
    vertPixelID = np.clip(vertPixelID, 0, CAMERA_PARA["height"] - 1)
    point_pixel_idx = np.array([horiPixelID, vertPixelID, pixelDepth]).T
    
    return point_pixel_idx

def scan2pixels_scannet(cloud):
    rgb_intrinsics = {
        'fx': 1169.621094,
        'fy': 1167.105103,
        'cx': 646.295044,
        'cy': 489.927032,
    }

    rgb_width = 1296
    rgb_height = 968

    x = cloud[:, 0]
    y = cloud[:, 1]
    x_rgb = x * rgb_intrinsics['fx'] / (cloud[:, 2] + 1e-6) + rgb_intrinsics['cx']
    y_rgb = y * rgb_intrinsics['fy'] / (cloud[:, 2] + 1e-6) + rgb_intrinsics['cy']

    point_pixel_idx = np.array([y_rgb, x_rgb, cloud[:, 2]]).T
    return point_pixel_idx


def scan2pixels_custom_seeker(laserCloud, fov_up, yaw_offset, z_offset):
    img_width = 1280   
    img_height = 640   
    # 严格同步 C++: fov_up_rad_
    fov_up_rad = fov_up * np.pi / 180.0 

    x = laserCloud[:, 0]
    y = laserCloud[:, 1]
    
    # 1. 严格同步 C++ L105: z = z + z_offset (z_offset 取 0.12)
    z = laserCloud[:, 2] + z_offset 

    # 2. 严格同步 C++ L107: 计算距离
    dist = np.sqrt(x*x + y*y + z*z)
    
    # 3. 严格同步 C++ L108: 距离过滤
    # 过滤掉 0.3m 以内(身体)和 25m 以外(噪声)
    valid_mask = (dist > 0.3) & (dist < 25.0)
    
    point_pixel_idx = np.full((laserCloud.shape[0], 3), -1.0)

    if np.any(valid_mask):
        xv, yv, zv, dv = x[valid_mask], y[valid_mask], z[valid_mask], dist[valid_mask]

        # 4. 严格同步 C++ L111-118: 方位角计算
        theta = -np.arctan2(yv, xv)
        theta = np.where(theta < 0, theta + 2.0 * np.pi, theta)
        theta += (yaw_offset * np.pi / 180.0)
        theta = theta % (2.0 * np.pi)

        # 5. 严格同步 C++ L121: 俯仰角 (必须是正 asin)
        phi = np.arcsin(np.clip(zv / dv, -1.0, 1.0))

        # 6. 严格同步 C++ L124-125: 像素映射 (必须是 0.5 - phi)
        # 逻辑：phi 为负(地面)时，-phi 为正，v > 0.5，点落在图像下方
        u = (theta / (2.0 * np.pi)) * img_width
        v = (0.5 - phi / (2.0 * fov_up_rad)) * img_height 

        point_pixel_idx[valid_mask, 0] = u
        point_pixel_idx[valid_mask, 1] = v
        point_pixel_idx[valid_mask, 2] = dv

    return point_pixel_idx



# # @profile
# import jax
# import jax.numpy as jnp

# @jax.jit
# def min_depth_per_pixel(coords, depths):
#     """
#     coords: (N, 2) int array -> each row is (x, y)
#     depths: (N,) float array -> depth for each pixel
    
#     Returns:
#     unique_coords: (M, 2) array of unique pixel coords
#     min_depths: (M,) array of minimum depth per unique coord
#     """
#     # 1) Get unique coordinates + inverse index
#     coords = jnp.array(coords)
#     depths = jnp.array(depths)

#     # unique_coords, inv_idx = jnp.unique(coords, axis=0, return_inverse=True, size=coords.shape[0], fill_value=-1)
    
#     # unique_coords = unique_coords[unique_coords[:, 0] != -1]  # Remove fill_value
#     # assert len(inv_idx) == len(coords)

#     # 2) Prepare output array for minimum depth
#     min_depths = jnp.full(len(coords), jnp.inf, dtype=depths.dtype) # discard the last element at last
    
#     # # 3) "Scatter" minimum using np.minimum.at
#     # #    This will, for each index in inv_idx, do:
#     # #       min_depths[inv_idx[i]] = min(min_depths[inv_idx[i]], depths[i])
#     # for i in range(len(coords)):
#     #     # min_depths[inv_idx[i]] = jnp.min(min_depths[inv_idx[i]], depths[i])
#     #     check_depth = min_depths[inv_idx[i]]
#     #     if depths[i] < check_depth:
#     #         min_depths = jax.ops.index_update(min_depths, inv_idx[i], depths[i])
            
#     #     # min_depths.at[inv_idx[i]].set(jnp.min(min_depths[inv_idx[i]], depths[i]))

#     def body_fun(i, current_min_depths):
#         # Get the current depth and the index corresponding to the coordinate.
#         current_depth = depths[i]
#         current_value = current_min_depths[i]
#         # Use jax.lax.select to pick the minimum without a Python if.
#         new_value = jax.lax.select(current_depth < current_value, current_depth, current_value)
#         return current_min_depths.at[i].set(new_value)
    
#     final_min_depths = jax.lax.fori_loop(0, len(coords), body_fun, min_depths)
    
#     # return unique_coords, final_min_depths
#     return coords, final_min_depths

class CloudImageFusion:
    def __init__(self, platform, fov_up,yaw_offset, z_offset):
        self.fov_up = fov_up
        self.yaw_offset=yaw_offset
        self.z_offset=z_offset
        self.platform = platform # 保存平台名称
        
        self.platform_list = ['wheelchair', 'mecanum', 'mecanum_sim', 'scannet', 'diablo', 'custom_seeker']
        if platform not in self.platform_list:
            raise ValueError(f"Invalid platform: {platform}")
    
    def scan2pixels(self, laserCloud):
        # 1. 处理 custom_seeker (带 3 个额外参数)
        if self.platform == 'custom_seeker':
            return scan2pixels_custom_seeker(
                laserCloud, 
                fov_up=self.fov_up, 
                yaw_offset=self.yaw_offset, 
                z_offset=self.z_offset
            )
        
        # 2. 处理 scannet
        if self.platform == 'scannet':
            return scan2pixels_scannet(laserCloud)
            
        # 3. 处理其他标准平台 (wheelchair, mecanum, mecanum_sim, diablo)
        # 使用 globals() 动态获取函数名并执行
        func_name = f"scan2pixels_{self.platform}"
        return globals()[func_name](laserCloud)
    
    # def generate_seg_cloud(self, cloud: np.ndarray, masks, labels, confidences, R_b2w, t_b2w, image_src=None):
    #     # Project the cloud points to image pixels
    #     point_pixel_idx = self.scan2pixels(cloud)

    #     if masks is None or len(masks) == 0:
    #         return None, None
        
    #     image_shape = masks[0].shape
        
    #     out_of_bound_filter = (point_pixel_idx[:, 0] >= 0) & \
    #                         (point_pixel_idx[:, 0] < image_shape[1]) & \
    #                         (point_pixel_idx[:, 1] >= 0) & \
    #                         (point_pixel_idx[:, 1] < image_shape[0])

    #     point_pixel_idx = point_pixel_idx[out_of_bound_filter]
    #     cloud = cloud[out_of_bound_filter]
        
    #     horDis = point_pixel_idx[:, 2]
    #     point_pixel_idx = point_pixel_idx.astype(int)

    #     all_obj_cloud_mask = np.zeros(cloud.shape[0], dtype=bool)
    #     obj_cloud_world_list = []
    #     for i in range(len(labels)):
    #         obj_mask = masks[i]
    #         cloud_mask = obj_mask[point_pixel_idx[:, 1], point_pixel_idx[:, 0]].astype(bool)
    #         all_obj_cloud_mask = np.logical_or(all_obj_cloud_mask, cloud_mask)
    #         obj_cloud = cloud[cloud_mask]
                    
    #         # obj_cloud_list.append(obj_cloud)
            
    #         obj_cloud_world = obj_cloud[:, :3] @ R_b2w.T + t_b2w
    #         obj_cloud_world_list.append(obj_cloud_world)
        
    #     if image_src is not None:
    #         # 1. 之前是根据距离变色，现在我们统一改为极其醒目的“亮紫色”或“亮黄色”
    #         # [B, G, R] 格式，亮紫色为 [255, 0, 255]
    #         high_contrast_color = [255, 0, 255] 
    #         all_obj_point_pixel_idx = point_pixel_idx

    #         # 2. 为了让颜色更深、更清晰，我们不再直接操作像素数组，而是画小圆点
    #         # 这样点与点之间会连起来，更容易看出偏移方向
    #         for i in range(len(all_obj_point_pixel_idx)):
    #             u = int(all_obj_point_pixel_idx[i, 0])
    #             v = int(all_obj_point_pixel_idx[i, 1])
    #             # cv2.circle(图像, 中心点, 半径, 颜色, -1表示实心)
    #             cv2.circle(image_src, (u, v), 2, high_contrast_color, -1)

    #     return obj_cloud_world_list

    #     # if image_src is not None:
    #     #     all_obj_cloud = cloud
    #     #     all_obj_point_pixel_idx = point_pixel_idx
    #     #     horDis = horDis
    #     #     # all_obj_cloud = cloud[all_obj_cloud_mask]
    #     #     # all_obj_point_pixel_idx = point_pixel_idx[all_obj_cloud_mask]
    #     #     # horDis = horDis[all_obj_cloud_mask]
    #     #     maxRange = 6.0
    #     #     pixelVal = np.clip(255 * horDis / maxRange, 0, 255).astype(np.uint8)
    #     #     image_src[all_obj_point_pixel_idx[:, 1], all_obj_point_pixel_idx[:, 0]] = np.array([pixelVal, 255-pixelVal, np.zeros_like(pixelVal)]).T # assume RGB
        
    #     # return obj_cloud_world_list

    def generate_seg_cloud(self, cloud: np.ndarray, masks, labels, confidences, R_b2w, t_b2w, image_src=None):
        # 1. 执行投影
        point_pixel_idx = self.scan2pixels(cloud)

        if masks is None or len(masks) == 0:
            return None, None
        
        image_shape = masks[0].shape
        
        # 2. 边界过滤 (同时过滤掉 scan2pixels 中标记为 -1 的无效点)
        out_of_bound_filter = (point_pixel_idx[:, 0] >= 0) & \
                            (point_pixel_idx[:, 0] < image_shape[1]) & \
                            (point_pixel_idx[:, 1] >= 0) & \
                            (point_pixel_idx[:, 1] < image_shape[0])

        point_pixel_idx = point_pixel_idx[out_of_bound_filter]
        cloud = cloud[out_of_bound_filter]
        
        # 预转为整数索引以提升后续掩码提取速度
        u_int = point_pixel_idx[:, 0].astype(int)
        v_int = point_pixel_idx[:, 1].astype(int)
        dist_vals = point_pixel_idx[:, 2]

        # 3. 语义掩码提取逻辑
        obj_cloud_world_list = []
        for i in range(len(labels)):
            obj_mask = masks[i]
            # 利用整数索引直接从 mask 矩阵提取布尔掩码
            cloud_mask = obj_mask[v_int, u_int].astype(bool)
            obj_cloud = cloud[cloud_mask]
            
            # 投影到世界坐标系
            obj_cloud_world = obj_cloud[:, :3] @ R_b2w.T + t_b2w
            obj_cloud_world_list.append(obj_cloud_world)
        
        # 4. 绘图调试部分：加深颜色并加大点径
        if image_src is not None:
            for i in range(len(point_pixel_idx)):
                u, v, d = u_int[i], v_int[i], dist_vals[i]
                
                # --- 高对比度 Jet 颜色映射逻辑 ---
                # 归一化距离到 [0, 1]，设定 12米为最大红色范围（室内调试更敏感）
                norm_d = np.clip(d / 12.0, 0, 1)
                
                # 颜色分段映射 (BGR 格式)
                if norm_d < 0.25: # 蓝 -> 青
                    b, g, r = 255, int(1020 * norm_d), 0
                elif norm_d < 0.5: # 青 -> 绿
                    b, g, r = int(255 - 1020 * (norm_d - 0.25)), 255, 0
                elif norm_d < 0.75: # 绿 -> 黄
                    b, g, r = 0, 255, int(1020 * (norm_d - 0.5))
                else: # 黄 -> 红
                    b, g, r = 0, int(255 - 1020 * (norm_d - 0.75)), 255
                
                # --- 深度增强绘制 ---
                # 将半径从 1 增加到 2 或 3，会显著加深视觉感官
                # thickness=-1 表示填充实心圆
                cv2.circle(image_src, (u, v), 2, (b, g, r), -1)

        return obj_cloud_world_list
    

    # @profile
    def generate_seg_cloud_v2(self, cloud: np.ndarray, masks, labels, confidences, R_b2w, t_b2w, image_src=None):
        point_pixel_idx = self.scan2pixels(cloud)

        if masks is None:
            return None, None
        
        image_shape = masks[0].shape
        
        out_of_bound_filter = (point_pixel_idx[:, 0] >= 0) & \
                            (point_pixel_idx[:, 0] < image_shape[1]) & \
                            (point_pixel_idx[:, 1] >= 0) & \
                            (point_pixel_idx[:, 1] < image_shape[0])

        point_pixel_idx = point_pixel_idx[out_of_bound_filter]
        cloud = cloud[out_of_bound_filter]
        
        depths = point_pixel_idx[:, 2]
        point_pixel_idx = point_pixel_idx.astype(int)

        depth_image = np.full(image_shape, np.inf, dtype=np.float32)

        import time
        start_time = time.time()

        # pixel_indices, depths = min_depth_per_pixel(point_pixel_idx[:, :2], horDis)
        # pixel_indices = np.array(pixel_indices, dtype=int)
        # pixel_indices = pixel_indices[pixel_indices[:, 0] >= 0]
        # depths = np.array(depths)

        np.minimum.at(depth_image, (point_pixel_idx[:, 1], point_pixel_idx[:, 0]), depths)
        structure = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=np.uint8)
        inflated_depth_image = scipy.ndimage.grey_dilation(depth_image, footprint=structure, mode='nearest')

        inflated_depth_image = np.minimum(inflated_depth_image, depth_image)

        print(f'pixel conversion: {time.time() - start_time} for {point_pixel_idx.shape[0]} points')
        # for i, pixel_idx in enumerate(pixel_indices):
        #     depth_image[*pixel_idx[[1, 0]].tolist()] = depths[i]
            
        # depth_image[pixel_indices[:, 1], pixel_indices[:, 0]] = depths

        valid_mask = ~np.isinf(inflated_depth_image)  # Mask for valid depth values
        if valid_mask.any():
            min_depth = inflated_depth_image[valid_mask].min()
            max_depth = inflated_depth_image[valid_mask].max()

            print(f"Min depth: {min_depth}, Max depth: {max_depth}")

            # Normalize only valid depth values
            normalized_depth = np.zeros_like(inflated_depth_image, dtype=np.uint8)
            normalized_depth[valid_mask] = 255 * (1 - (inflated_depth_image[valid_mask] - min_depth) / (max_depth - min_depth + 1e-6))
        else:
            normalized_depth = np.zeros_like(inflated_depth_image, dtype=np.uint8)  # If all values are inf, return a blank image
        
        # cv2.imshow("Depth Image", normalized_depth)
        # cv2.waitKey(1)  # Wait for a key press to close the window

        all_obj_cloud_mask = np.zeros(cloud.shape[0], dtype=bool)
        obj_cloud_world_list = []
        for i in range(len(labels)):
            obj_mask = masks[i]
            cloud_mask = obj_mask[point_pixel_idx[:, 1], point_pixel_idx[:, 0]].astype(bool)
            all_obj_cloud_mask = np.logical_or(all_obj_cloud_mask, cloud_mask)
            obj_cloud = cloud[cloud_mask]
                    
            # obj_cloud_list.append(obj_cloud)
            
            obj_cloud_world = obj_cloud[:, :3] @ R_b2w.T + t_b2w
            obj_cloud_world_list.append(obj_cloud_world)

        if image_src is not None:
            all_obj_cloud = cloud
            all_obj_point_pixel_idx = point_pixel_idx
            horDis = horDis
            # all_obj_cloud = cloud[all_obj_cloud_mask]
            # all_obj_point_pixel_idx = point_pixel_idx[all_obj_cloud_mask]
            # horDis = horDis[all_obj_cloud_mask]
            maxRange = 6.0
            pixelVal = np.clip(255 * horDis / maxRange, 0, 255).astype(np.uint8)
            image_src[all_obj_point_pixel_idx[:, 1], all_obj_point_pixel_idx[:, 0]] = np.array([pixelVal, 255-pixelVal, np.zeros_like(pixelVal)]).T # assume RGB
        
        return obj_cloud_world_list, normalized_depth
