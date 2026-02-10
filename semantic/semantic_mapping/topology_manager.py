import rclpy
from rclpy.node import Node
import numpy as np
from scipy.spatial import KDTree
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped, Point 
from sensor_msgs.msg import PointCloud2
import networkx as nx
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

class TopologyManager(Node):
    def __init__(self):
        super().__init__('topology_manager')
        
        # 订阅
        self.pose_sub = self.create_subscription(PoseStamped, '/mavros/vision_pose/pose', self.pose_callback, 10)
        self.cloud_sub = self.create_subscription(PointCloud2, '/cloud_registered', self.cloud_callback, 10)
        self.obj_sub = self.create_subscription(MarkerArray, '/obj_boxes', self.obj_callback, 10)
        self.viz_pub = self.create_publisher(MarkerArray, '/scene_graph_viz', 10)

        # 参数与常量
        self.HEIGHT_PLACES = 1.2
        self.HEIGHT_ROOMS = 3.5
        self.HEIGHT_BUILDING = 5.5
        self.min_dist_between_nodes = 1.2
        self.room_threshold = 0.7 # 判定房间核心的半径阈值

        self.color_palette = [
            ColorRGBA(r=1.0, g=0.2, b=0.2, a=0.9), ColorRGBA(r=0.2, g=1.0, b=0.2, a=0.9),
            ColorRGBA(r=0.2, g=0.5, b=1.0, a=0.9), ColorRGBA(r=1.0, g=0.8, b=0.0, a=0.9),
            ColorRGBA(r=0.8, g=0.2, b=1.0, a=0.9), ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.9),
        ]

        # 状态变量
        self.latest_cloud_msg = None
        self.last_pos = None
        self.graph = nx.Graph()
        self.node_count = 0
        self.current_objects = {} # {id: {pos, label}}
        
        self.analysis_timer = self.create_timer(5.0, self.graph_analysis_callback)
        self.room_id_to_color = {}

    def cloud_callback(self, msg):
        self.latest_cloud_msg = msg

    def obj_callback(self, msg):
        for marker in msg.markers:
            if marker.action == 0:  # ADD/MODIFY
                obj_id = f"obj_{marker.id}"
                pos = np.array([marker.pose.position.x, marker.pose.position.y, marker.pose.position.z])
                self.current_objects[obj_id] = {'pos': pos, 'label': marker.ns}
                
                if not self.graph.has_node(obj_id):
                    self.graph.add_node(obj_id, type='object', label=marker.ns, pos=pos)
                else:
                    self.graph.nodes[obj_id]['pos'] = pos # 更新位置

                # 建立与附近地点的连接
                for node, data in self.graph.nodes(data=True):
                    if data.get('type') == 'place':
                        if np.linalg.norm(pos - data['pos']) < 3.0:
                            self.graph.add_edge(obj_id, node)

    def pose_callback(self, msg):
        curr_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        if self.last_pos is None or np.linalg.norm(curr_pos - self.last_pos) > self.min_dist_between_nodes:
            if self.latest_cloud_msg is not None:
                # 无论是否生成节点成功，都先记录位置防止死循环
                self.process_and_generate_node(curr_pos)
                self.last_pos = curr_pos 
            else:
                self.get_logger().warn("等待点云数据...")

    def process_and_generate_node(self, curr_pos):
        try:
            points = pc2.read_points_numpy(self.latest_cloud_msg, field_names=("x", "y", "z"))
            if len(points) > 0:
                tree = KDTree(points)
                dist, _ = tree.query(curr_pos)
                
                # 放宽要求，只要离墙 10cm 以上就生成节点，保证路径连续
                if dist > 0.1: 
                    new_place_id = f"p_{self.node_count}"
                    self.graph.add_node(new_place_id, pos=curr_pos, radius=dist, type='place')
                    
                    # 连接上一个地点
                    if self.node_count > 0:
                        prev_id = f"p_{self.node_count-1}"
                        if self.graph.has_node(prev_id):
                            self.graph.add_edge(new_place_id, prev_id)
                    
                    # 关联物体
                    self.link_objects_to_place(new_place_id, curr_pos, dist)
                    self.node_count += 1
                    self.get_logger().info(f"📍 生成 Place {new_place_id} (R={dist:.2f}m)")
        except Exception as e:
            self.get_logger().error(f"生成节点失败: {e}")

    def link_objects_to_place(self, place_id, place_pos, radius):
        for obj_id, data in self.current_objects.items():
            dist = np.linalg.norm(data['pos'] - place_pos)
            if dist < max(radius, 3.0): 
                self.graph.add_edge(obj_id, place_id)

    def graph_analysis_callback(self):
        place_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'place']
        if not place_nodes: return
        
        edges_to_remove = []
        for u, v in self.graph.edges():
            type_u = self.graph.nodes[u].get('type')
            type_v = self.graph.nodes[v].get('type')
            # 如果边连接了房间层级，就标记删除
            if 'room' in [type_u, type_v] or 'building' in [type_u, type_v]:
                edges_to_remove.append((u, v))
        self.graph.remove_edges_from(edges_to_remove)

        # 1. 提取房间核心 (模拟“放气”)
        wide_nodes = [n for n in place_nodes if self.graph.nodes[n].get('radius', 0) > self.room_threshold]
        room_cores = list(nx.connected_components(self.graph.subgraph(wide_nodes))) if wide_nodes else []

        # 2. 兜底划分
        if not room_cores:
            room_cores = list(nx.connected_components(self.graph.subgraph(place_nodes)))
            self.get_logger().info(f"🏠 初始阶段：创建 {len(room_cores)} 个基础区域")

        # 3. 更新图中的 Room 节点和映射
        node_to_room = {}
        for i, core in enumerate(room_cores):
            room_id = f"room_{i}"
            avg_pos = np.mean([self.graph.nodes[p]['pos'] for p in core], axis=0)
            
            if not self.graph.has_node(room_id):
                self.graph.add_node(room_id, type='room', pos=avg_pos)
            else:
                self.graph.nodes[room_id]['pos'] = avg_pos
            
            for p_id in core:
                node_to_room[p_id] = room_id
                self.graph.add_edge(p_id, room_id)

        # 4. 将物体和窄点吸附到最近房间
        for n, d in self.graph.nodes(data=True):
            if d['type'] in ['place', 'object'] and n not in node_to_room:
                # 寻找已经有房间归属的最近邻居
                for neighbor in self.graph.neighbors(n):
                    if neighbor in node_to_room:
                        self.graph.add_edge(n, node_to_room[neighbor])
                        break

        # 5. Building (L5)
        room_ids = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'room']
        if room_ids:
            b_id = "b_0"
            if not self.graph.has_node(b_id): self.graph.add_node(b_id, type='building')
            b_pos = np.mean([self.graph.nodes[r]['pos'] for r in room_ids], axis=0)
            self.graph.nodes[b_id]['pos'] = b_pos
            for r in room_ids: self.graph.add_edge(b_id, r)

        self.publish_graph_to_rviz()

    def get_node_viz_pos(self, node_id):
        if node_id not in self.graph: return None
        d = self.graph.nodes[node_id]
        pos = d.get('pos', np.array([0., 0., 0.]))
        p = Point(x=float(pos[0]), y=float(pos[1]))
        if d['type'] == 'building': p.z = self.HEIGHT_BUILDING
        elif d['type'] == 'room': p.z = self.HEIGHT_ROOMS
        elif d['type'] == 'place': p.z = self.HEIGHT_PLACES
        else: p.z = float(pos[2])
        return p

    def publish_graph_to_rviz(self):
        ma = MarkerArray()
        now = self.get_clock().now().to_msg()
        
        # 建立 房间->颜色 映射
        room_ids = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'room']
        # 确保每个房间 ID 都有固定的颜色
        for rid in room_ids:
            if rid not in self.room_id_to_color:
                # 为新发现的房间 ID 分配调色盘中下一个可用的颜色
                color_idx = len(self.room_id_to_color) % len(self.color_palette)
                self.room_id_to_color[rid] = self.color_palette[color_idx]

        # 使用持久化的映射表
        r_colors = self.room_id_to_color
                
        # 地点归属映射
        p_to_r = {}
        for rid in room_ids:
            for nbr in self.graph.neighbors(rid):
                if self.graph.nodes[nbr].get('type') == 'place': p_to_r[nbr] = rid

        for n, d in self.graph.nodes(data=True):
            marker = Marker()
            marker.header.frame_id, marker.header.stamp = "map", now
            marker.ns, marker.id = d['type'], hash(n) % 2147483647
            marker.action = Marker.ADD
            vpos = self.get_node_viz_pos(n)
            marker.pose.position = vpos
            
            if d['type'] == 'building':
                marker.type, marker.scale.x = Marker.CUBE, 0.6
                marker.scale.y = marker.scale.z = 0.6
                marker.color = ColorRGBA(r=0.4, g=0.0, b=0.4, a=1.0)
            elif d['type'] == 'room':
                marker.type, marker.scale.x = Marker.CUBE, 0.4
                marker.scale.y = marker.scale.z = 0.4
                marker.color = r_colors.get(n, ColorRGBA(r=1.0, a=1.0))
            elif d['type'] == 'place':
                marker.type, marker.scale.x = Marker.SPHERE, 0.25
                marker.scale.y = marker.scale.z = 0.25
                rid = p_to_r.get(n)
                marker.color = r_colors[rid] if rid else ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.8)
            else: # object
                marker.type, marker.scale.x = Marker.SPHERE, 0.15
                marker.scale.y = marker.scale.z = 0.15
                marker.color = ColorRGBA(r=0.2, g=0.2, b=0.2, a=0.5)
            ma.markers.append(marker)

        # 连线
        line = Marker(type=Marker.LINE_LIST, ns="edges", id=0, action=Marker.ADD)
        line.header.frame_id, line.header.stamp = "map", now
        line.scale.x = 0.02
        for u, v in self.graph.edges():
            p1, p2 = self.get_node_viz_pos(u), self.get_node_viz_pos(v)
            if p1 and p2:
                line.points.extend([p1, p2])
                tu, tv = self.graph.nodes[u]['type'], self.graph.nodes[v]['type']
                if 'building' in [tu, tv]: c = ColorRGBA(r=0.0, g=0.0, b=0.0, a=0.6)
                elif tu == 'room': c = r_colors.get(u, ColorRGBA(a=0.1))
                elif tv == 'room': c = r_colors.get(v, ColorRGBA(a=0.1))
                else: c = ColorRGBA(r=0.8, g=0.8, b=0.8, a=0.1)
                line.colors.extend([c, c])
        ma.markers.append(line)
        self.viz_pub.publish(ma)

def main(args=None):
    rclpy.init(args=args)
    node = TopologyManager()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()