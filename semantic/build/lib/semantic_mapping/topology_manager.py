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
            # 1. 过滤掉无效的 Marker (类型0通常是占位或删除信号)
            if marker.type == 0 or marker.action != 0: 
                continue

            obj_id = f"obj_{marker.ns}_{marker.id}" 
            
            # 2. 【核心修复】判断 Marker 类型并提取真实坐标
            if marker.type == 5:  # LINE_LIST (你的物体框类型)
                if len(marker.points) > 0:
                    # 计算 24 个顶点的平均值作为物体的中心坐标
                    pts = np.array([[p.x, p.y, p.z] for p in marker.points])
                    pos = np.mean(pts, axis=0)
                else:
                    continue
            else:
                # 其他类型（如 CUBE/SPHERE）通常直接用 pose.position
                pos = np.array([marker.pose.position.x, marker.pose.position.y, marker.pose.position.z])

            # 3. 检查算出来的坐标是否依然为 0 (防止异常)
            if np.all(pos == 0):
                continue

            # --- 以下保持你的逻辑不变 ---
            self.current_objects[obj_id] = {'pos': pos, 'label': marker.ns}
            
            if not self.graph.has_node(obj_id):
                self.graph.add_node(obj_id, type='object', label=marker.ns, pos=pos)
            else:
                self.graph.nodes[obj_id]['pos'] = pos

            # 打印真实坐标进行验证
            # self.get_logger().info(f"📦 [Object] {marker.ns}({marker.id}) 真实坐标: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")


    def pose_callback(self, msg):
        curr_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        if self.last_pos is None or np.linalg.norm(curr_pos - self.last_pos) > self.min_dist_between_nodes:
            if self.latest_cloud_msg is not None:
                # 无论是否生成节点成功，都先记录位置防止死循环
                self.process_and_generate_node(curr_pos)
                self.last_pos = curr_pos 
            else:
                self.get_logger().warn("等待点云数据...")

    # def process_and_generate_node(self, curr_pos):
    #     try:
    #         points = pc2.read_points_numpy(self.latest_cloud_msg, field_names=("x", "y", "z"))
    #         if len(points) > 0:
    #             tree = KDTree(points)
    #             dist, _ = tree.query(curr_pos)
                
    #             # 放宽要求，只要离墙 10cm 以上就生成节点，保证路径连续
    #             if dist > 0.1: 
    #                 new_place_id = f"p_{self.node_count}"
    #                 self.graph.add_node(new_place_id, pos=curr_pos, radius=dist, type='place')
                    
    #                 # 连接上一个地点
    #                 if self.node_count > 0:
    #                     prev_id = f"p_{self.node_count-1}"
    #                     if self.graph.has_node(prev_id):
    #                         self.graph.add_edge(new_place_id, prev_id)
                    
    #                 # 关联物体
    #                 self.link_objects_to_place(new_place_id, curr_pos, dist)
    #                 self.node_count += 1
    #                 if self.node_count == 1:
    #                     self.graph_analysis_callback()
    #                 self.get_logger().info(f"📍 生成 Place {new_place_id} (R={dist:.2f}m)")
    #     except Exception as e:
    #         self.get_logger().error(f"生成节点失败: {e}")
    def seek_gvd_center(self, start_pos, tree):
        """
        GVD 思想实现：寻找局部最大距离点
        """
        best_pos = start_pos
        max_dist, _ = tree.query(start_pos)
        
        # 定义探测方向 (米)：上下左右及斜向偏移
        offsets = [
            [0.5, 0, 0], [-0.5, 0, 0], [0, 0.5, 0], [0, -0.5, 0],
            [0.35, 0.35, 0], [-0.35, -0.35, 0]
        ]
        
        for off in offsets:
            candidate_pos = start_pos + np.array(off)
            d, _ = tree.query(candidate_pos)
            # 如果探测点离墙更远，说明更接近“中轴线”
            if d > max_dist:
                max_dist = d
                best_pos = candidate_pos
                
        return best_pos, max_dist
    
    def process_and_generate_node(self, curr_pos):
        try:
            points = pc2.read_points_numpy(self.latest_cloud_msg, field_names=("x", "y", "z"))
            if len(points) > 0:
                tree = KDTree(points)
                
                # --- GVD 核心逻辑：寻找局部“最空旷”点 ---
                # 不直接用当前点，而是在周围探测一下，找一个离墙最远的位置
                gvd_pos, dist = self.seek_gvd_center(curr_pos, tree)
                
                if dist > 0.1: 
                    new_place_id = f"p_{self.node_count}"
                    # 使用优化后的 gvd_pos 而不是原始的 curr_pos
                    self.graph.add_node(new_place_id, pos=gvd_pos, radius=dist, type='place')
                    # self.get_logger().info(f"📍 [Place] {new_place_id} 坐标: x={gvd_pos[0]:.2f}, y={gvd_pos[1]:.2f}, z={gvd_pos[2]:.2f}")
                    
                    if self.node_count > 0:
                        prev_id = f"p_{self.node_count-1}"
                        if self.graph.has_node(prev_id):
                            self.graph.add_edge(new_place_id, prev_id)
                            # self.get_logger().info(f"📍 [Place] {new_place_id} 坐标: x={gvd_pos[0]:.2f}, y={gvd_pos[1]:.2f}, z={gvd_pos[2]:.2f}")
                    
                    self.link_objects_to_place(new_place_id, gvd_pos, dist)
                    self.node_count += 1
                    
                    # 触发“开局即显示”
                    if self.node_count == 1:
                        self.graph_analysis_callback()
                        
                    self.get_logger().info(f"📍 GVD节点 {new_place_id} (R={dist:.2f}m)")
        except Exception as e:
            self.get_logger().error(f"GVD处理失败: {e}")

    def link_objects_to_place(self, place_id, place_pos, radius):
        for obj_id, data in self.current_objects.items():
            dist = np.linalg.norm(data['pos'] - place_pos)
            if dist < max(radius, 3.0): 
                self.graph.add_edge(obj_id, place_id)
                # # 新增 DEBUG 输出
                # label = data.get('label', 'unknown')
                # self.get_logger().info(f"🔗 [Reactive] 物体 {label}({obj_id}) 已连接新地点 {place_id}")

    # def graph_analysis_callback(self):
    #     place_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'place']
    #     if not place_nodes: return
        
    #     edges_to_remove = []
    #     for u, v in self.graph.edges():
    #         type_u = self.graph.nodes[u].get('type')
    #         type_v = self.graph.nodes[v].get('type')
    #         # 如果边连接了房间层级，就标记删除
    #         if 'room' in [type_u, type_v] or 'building' in [type_u, type_v]:
    #             edges_to_remove.append((u, v))
    #     self.graph.remove_edges_from(edges_to_remove)

    #     # 1. 提取房间核心 (模拟“放气”)
    #     wide_nodes = [n for n in place_nodes if self.graph.nodes[n].get('radius', 0) > self.room_threshold]
    #     room_cores = list(nx.connected_components(self.graph.subgraph(wide_nodes))) if wide_nodes else []

    #     # 2. 兜底划分
    #     if not room_cores:
    #         room_cores = list(nx.connected_components(self.graph.subgraph(place_nodes)))
    #         self.get_logger().info(f"🏠 初始阶段：创建 {len(room_cores)} 个基础区域")

    #     # 3. 更新图中的 Room 节点和映射
    #     node_to_room = {}
    #     for i, core in enumerate(room_cores):
    #         room_id = f"room_{i}"
    #         avg_pos = np.mean([self.graph.nodes[p]['pos'] for p in core], axis=0)
            
    #         if not self.graph.has_node(room_id):
    #             self.graph.add_node(room_id, type='room', pos=avg_pos)
    #         else:
    #             self.graph.nodes[room_id]['pos'] = avg_pos
            
    #         for p_id in core:
    #             node_to_room[p_id] = room_id
    #             self.graph.add_edge(p_id, room_id)

    #     # 4. 将物体和窄点吸附到最近房间
    #     for n, d in self.graph.nodes(data=True):
    #         if d['type'] in ['place', 'object'] and n not in node_to_room:
    #             # 寻找已经有房间归属的最近邻居
    #             for neighbor in self.graph.neighbors(n):
    #                 if neighbor in node_to_room:
    #                     self.graph.add_edge(n, node_to_room[neighbor])
    #                     break

    #     # 5. Building (L5)
    #     room_ids = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'room']
    #     if room_ids:
    #         b_id = "b_0"
    #         if not self.graph.has_node(b_id): self.graph.add_node(b_id, type='building')
    #         b_pos = np.mean([self.graph.nodes[r]['pos'] for r in room_ids], axis=0)
    #         self.graph.nodes[b_id]['pos'] = b_pos
    #         for r in room_ids: self.graph.add_edge(b_id, r)

    #     self.publish_graph_to_rviz()

    def reconcile_object_to_places(self):
        """
        全局对齐：确保每个物体【仅连接】一个最近的地点
        """
        obj_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'object']
        place_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'place']
        
        if not obj_nodes or not place_nodes: return

        # 提取地点坐标
        p_ids = []
        p_coords = []
        for p in place_nodes:
            p_ids.append(p)
            p_coords.append(self.graph.nodes[p]['pos'])
        p_coords = np.array(p_coords)

        for o_id in obj_nodes:
            # --- 核心改进：先删除该物体现有的所有【地点】连接 ---
            current_neighbors = list(self.graph.neighbors(o_id))
            for nbr in current_neighbors:
                if self.graph.nodes[nbr].get('type') == 'place':
                    self.graph.remove_edge(o_id, nbr)

            # --- 寻找最近的唯一地点 ---
            o_pos = self.graph.nodes[o_id]['pos']
            dists = np.linalg.norm(p_coords - o_pos, axis=1)
            min_idx = np.argmin(dists)
            
            # 只建立这一条最短的边
            self.graph.add_edge(o_id, p_ids[min_idx])

    def generate_hierarchy_description(self):
        lines = ["Current Scene Hierarchy:"]
        room_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'room']
        for r_id in room_nodes:
            objects_in_room = []
            # 找到属于该房间的所有地点
            associated_places = [n for n in self.graph.neighbors(r_id) if self.graph.nodes[n].get('type') == 'place']
            for p_id in associated_places:
                # 找到连向该地点的所有物体
                for neighbor in self.graph.neighbors(p_id):
                    if self.graph.nodes[neighbor].get('type') == 'object':
                        label = self.graph.nodes[neighbor].get('label', 'unknown')
                        objects_in_room.append(f"{label}")
            
            lines.append(f"- {r_id}: Contains {list(set(objects_in_room))}") # 去重显示
        return "\n".join(lines)
    
    def anti_neck_merge(self, cores, place_nodes, delta):
        """
        cores: List[Set[place_id]]
        """
        if len(cores) <= 1:
            return cores

        merged = []
        used = set()

        for i, core_a in enumerate(cores):
            if i in used:
                continue

            merged_core = set(core_a)

            for j, core_b in enumerate(cores):
                if j <= i or j in used:
                    continue

                # --- 原始 place 图是否连通 ---
                connected = False
                bridge_count = float('inf')

                for pa in core_a:
                    for pb in core_b:
                        if nx.has_path(self.graph, pa, pb):
                            path = nx.shortest_path(self.graph, pa, pb)
                            # 计算“被过滤掉”的节点数量
                            bridge_nodes = [
                                p for p in path
                                if self.graph.nodes[p].get('radius', 0) <= delta
                            ]
                            bridge_count = min(bridge_count, len(bridge_nodes))
                            connected = True

                if not connected:
                    continue

                # --- cluster 距离 ---
                pos_a = np.mean([self.graph.nodes[p]['pos'] for p in core_a], axis=0)
                pos_b = np.mean([self.graph.nodes[p]['pos'] for p in core_b], axis=0)
                dist = np.linalg.norm(pos_a - pos_b)

                # --- 抗细脖子判据 ---
                if bridge_count <= 2 and dist < 4.0:
                    merged_core |= core_b
                    used.add(j)

            merged.append(merged_core)
            used.add(i)

        return merged

    
    def graph_analysis_callback(self):
        # 1. 首先确保物体和地点已经连上
        self.reconcile_object_to_places()

        # 2. 之后再执行原有的房间划分逻辑...
        place_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'place']
        if len(place_nodes) < 1: return

        # --- 1. 模拟持续同调 (Persistent Homology) ---
        thresholds = np.arange(0.3, 1.2, 0.02)
        betti_0_curve = [] # 记录每个阈值对应的区域数量
        clusters_at_threshold = {}

        for delta in thresholds:
            # 过滤：只保留半径 > delta 的节点（模拟放气）
            wide_nodes = [n for n in place_nodes if self.graph.nodes[n].get('radius', 0) > delta]
            if not wide_nodes:
                betti_0_curve.append(0)
                continue
            
            # 构建过滤后的子图并计算连通分量
            subgraph = self.graph.subgraph(wide_nodes)
            clusters = list(nx.connected_components(subgraph))
            betti_0_curve.append(len(clusters))
            clusters_at_threshold[delta] = clusters

        # --- 2. 寻找最稳定的“寿命” (Finding the stable plateau) ---
        stats = {} # 格式: {房间数量: 持续的长度}
        
        current_val = -1
        current_len = 0
        for count in betti_0_curve:
            if count == current_val:
                current_len += 1
            else:
                if current_val >= 1:
                    # 记录该房间数出现的最大连续长度
                    stats[current_val] = max(stats.get(current_val, 0), current_len)
                current_val = count
                current_len = 1
        # 处理最后一个序列
        if current_val >= 1:
            stats[current_val] = max(stats.get(current_val, 0), current_len)

        # 【决策逻辑】
        winning_count = 1
        multi_room_options = [c for c in stats.keys() if c > 1]
        
        if multi_room_options:
            # 找到持续步长最长的多房间方案
            best_multi_room = max(multi_room_options, key=lambda c: stats[c])
            # 如果这个方案能维持至少 2 个步长（即 0.04m 的范围），就采用它
            if stats[best_multi_room] >= 2:
                winning_count = best_multi_room

        # 找到该数量对应的最优阈值
        try:
            optimal_delta = thresholds[betti_0_curve.index(winning_count)]
        except:
            optimal_delta = thresholds[0] # 兜底选搜索范围的起点

        self.get_logger().info(f"📈 拓扑分析：自适应阈值选择 {optimal_delta:.1f}m，判定房间数：{winning_count}")

        # --- 3. 执行最终划分与清理 ---
        # 在应用新划分前，清理旧的 Room 和 Building 边
        self.clear_hierarchical_edges()

        raw_cores = clusters_at_threshold.get(optimal_delta, [set(place_nodes)])

        final_cores = self.anti_neck_merge(
            raw_cores,
            place_nodes,
            optimal_delta
        )
        
        node_to_room = {}
        for i, core in enumerate(final_cores):
            room_id = f"room_{i}"
            # 计算房间中心
            core_pos = [self.graph.nodes[p]['pos'] for p in core]
            avg_pos = np.mean(core_pos, axis=0)
            
            if not self.graph.has_node(room_id):
                self.graph.add_node(room_id, type='room', pos=avg_pos)
            else:
                self.graph.nodes[room_id]['pos'] = avg_pos

            for p_id in core:
                node_to_room[p_id] = room_id
                self.graph.add_edge(p_id, room_id)

        unassigned_places = [n for n in place_nodes if n not in node_to_room]
        
        changed = True
        while changed:
            changed = False
            for p_id in unassigned_places:
                if p_id not in node_to_room:
                    for neighbor in self.graph.neighbors(p_id):
                        if neighbor in node_to_room:
                            target_room = node_to_room[neighbor]
                            self.graph.add_edge(p_id, target_room)
                            node_to_room[p_id] = target_room # 关键：更新字典！
                            changed = True
                            break


        # --- 5. 顶层 Building 关联 ---
        self.update_building_layer()
        for n, d in self.graph.nodes(data=True):
            p = d.get('pos', [0,0,0])
            print(f"Node: {n} ({d['type']}) -> Pos: {p}")
        hierarchy_description = self.generate_hierarchy_description()
        self.get_logger().info(f"--- DSG Hierarchy Description ---\n{hierarchy_description}\n--------------------------------")
        self.publish_graph_to_rviz()

    def clear_hierarchical_edges(self):
        """清理旧的跨层边，防止线条杂乱"""
        edges_to_remove = []
        for u, v in self.graph.edges():
            types = [self.graph.nodes[u].get('type'), self.graph.nodes[v].get('type')]
            if 'room' in types or 'building' in types:
                edges_to_remove.append((u, v))
        self.graph.remove_edges_from(edges_to_remove)

    def update_building_layer(self):
        room_ids = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'room']
        if room_ids:
            if not self.graph.has_node("b_0"):
                self.graph.add_node("b_0", type='building')
            b_pos = np.mean([self.graph.nodes[r]['pos'] for r in room_ids], axis=0)
            self.graph.nodes["b_0"]['pos'] = b_pos
            for r in room_ids:
                self.graph.add_edge("b_0", r)

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
        
        # 1. 基础映射准备 (保持原有逻辑)
        room_ids = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'room']
        for rid in room_ids:
            if rid not in self.room_id_to_color:
                color_idx = len(self.room_id_to_color) % len(self.color_palette)
                self.room_id_to_color[rid] = self.color_palette[color_idx]

        r_colors = self.room_id_to_color
        p_to_r = {}
        for rid in room_ids:
            for nbr in self.graph.neighbors(rid):
                if self.graph.nodes[nbr].get('type') == 'place': 
                    p_to_r[nbr] = rid

        # 2. 绘制节点 (保持原有逻辑，Place 颜色已由 p_to_r 决定)
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
                marker.type, marker.scale.x = Marker.SPHERE, 0.1
                marker.scale.y = marker.scale.z = 0.1
                # 可选：让物体本身也带上颜色，如果不想要深灰，可以参考下方连线逻辑
                marker.color = ColorRGBA(r=0.85, g=0.85, b=0.85, a=0.1) 
            ma.markers.append(marker)

        # 3. 改进的连线着色逻辑
        line = Marker(type=Marker.LINE_LIST, ns="edges", id=0, action=Marker.ADD)
        line.header.frame_id, line.header.stamp = "map", now
        line.scale.x = 0.02
        
        for u, v in self.graph.edges():
            p1, p2 = self.get_node_viz_pos(u), self.get_node_viz_pos(v)
            if p1 and p2:
                line.points.extend([p1, p2])
                
                tu, tv = self.graph.nodes[u]['type'], self.graph.nodes[v]['type']
                c = ColorRGBA(r=0.8, g=0.8, b=0.8, a=0.1) # 默认淡灰色

                # --- 核心改进逻辑开始 ---
                
                # 情况 1: Building - Room 连线 (深色)
                if 'building' in [tu, tv]:
                    c = ColorRGBA(r=0.0, g=0.0, b=0.0, a=0.8)

                # 情况 2: Room - Place 连线
                elif tu == 'room' or tv == 'room':
                    rid = u if tu == 'room' else v
                    base_c = r_colors.get(rid)
                    # 关键：创建一个新对象，不要修改 base_c
                    c = ColorRGBA(r=base_c.r, g=base_c.g, b=base_c.b, a=0.8) 


                # 情况 3: Object - Place 连线 (重点改进)
                elif ('object' in [tu, tv]) and ('place' in [tu, tv]):
                    place_id = u if tu == 'place' else v
                    rid = p_to_r.get(place_id)
                    if rid:
                        base_c = r_colors[rid]
                        c = ColorRGBA(r=base_c.r, g=base_c.g, b=base_c.b, a=0.7)
                    else:
                        c = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.5)

                # 情况 4: Place - Place 连线 (同房同色)
                elif tu == 'place' and tv == 'place':
                    rid_u = p_to_r.get(u)
                    rid_v = p_to_r.get(v)
                    if rid_u and rid_u == rid_v: # 在同一个房间内
                        c = ColorRGBA(r=r_colors[rid_u].r, g=r_colors[rid_u].g, b=r_colors[rid_u].b, a=0.15)
                    else:
                        c = ColorRGBA(r=0.7, g=0.7, b=0.7, a=0.1)

                # --- 核心改进逻辑结束 ---
                
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