#!/usr/bin/env python3
import math
import heapq
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist, Point
from sensor_msgs.msg import LaserScan, Image as RosImage
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from tf_transformations import euler_from_quaternion

import cv2
from cv_bridge import CvBridge


def get_yaw_from_quaternion(q):
    quat = [q.x, q.y, q.z, q.w]
    _, _, yaw = euler_from_quaternion(quat)
    return yaw


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def sat(v, lo, hi):
    return max(lo, min(hi, v))


class GraphNode:
    def __init__(self, name: str):
        self.name = name
        self.neighbors = []
        self.costs = []

    def add_edges(self, nums, ws):
        self.neighbors.extend(nums)
        self.costs.extend(ws)


class GridGraph:
    def __init__(self):
        self.nodes = {}


class AStar:
    def __init__(self, graph: GridGraph):
        self.graph = graph
        self.g = {}
        self.h = {}
        self.prev = {}

    def _reset(self):
        self.g = {n: float("inf") for n in self.graph.nodes}
        self.h = {n: 0.0 for n in self.graph.nodes}
        self.prev = {n: None for n in self.graph.nodes}

    def _calc_heuristic(self, goal_name: str):
        gr, gc = map(int, goal_name.split(","))
        for name in self.graph.nodes:
            r, c = map(int, name.split(","))
            self.h[name] = math.hypot(gr - r, gc - c)

    def plan(self, start_name: str, goal_name: str):
        if start_name not in self.graph.nodes or goal_name not in self.graph.nodes:
            return False

        self._reset()
        self._calc_heuristic(goal_name)
        self.g[start_name] = 0.0

        open_heap = [(self.h[start_name], start_name)]
        closed = set()

        while open_heap:
            _, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            closed.add(cur)

            if cur == goal_name:
                return True

            node = self.graph.nodes[cur]
            for neighbor, w in zip(node.neighbors, node.costs):
                nb = neighbor.name
                newg = self.g[cur] + float(w)
                if newg < self.g[nb]:
                    self.g[nb] = newg
                    self.prev[nb] = cur
                    heapq.heappush(open_heap, (newg + self.h[nb], nb))

        return False

    def reconstruct(self, start_name: str, goal_name: str):
        if self.g.get(goal_name, float("inf")) == float("inf"):
            return []
        out = []
        cur = goal_name
        while cur is not None:
            out.append(cur)
            cur = self.prev[cur]
        out.reverse()
        return out


class RRTStarNode:
    def __init__(self, x, y, parent=None, cost=0.0):
        self.x = x
        self.y = y
        self.parent = parent
        self.cost = cost


class Task3Navigator(Node):
 

    def __init__(self):
        super().__init__("task3_algorithm")

        self.waypoints = [
            (-3.8455, -4.0161), (-4.0395, -1.6440), (-4.2054,  1.0219), (-4.0101,  3.1211),
            (-2.2667,  1.9064), ( 0.9358,  3.5652), (-0.5810, -0.5354), ( 3.0497, -0.6756),
            ( 5.2999,  2.1251), ( 2.9436,  3.0556), ( 3.8286,  0.8131), ( 8.5444,  3.0766),
            ( 8.6116, -0.1740), ( 8.9751, -2.7145),
            ( 8.0022, -5.6312), ( 0.1117,  0.0446),
        ]
        self.wp_idx = 0

        self.treat_unknown_as_occupied = True
        self.robot_radius = 0.19

        # Hard clearance inflation
        self.extra_inflation = 0.10  # meters

        # Soft clearance bias
        self.use_soft_clearance = True
        self.soft_clearance_m = 0.55
        self.soft_clearance_gain = 1.8

        # Pure pursuit
        self.lookahead = 0.28
        self.goal_tol = 0.22
        self.max_lin = 0.20
        self.max_ang = 0.85
        self.curvature_speed_k = 0.9
        self.min_lin_scale = 0.25

        # Scan slowdown only
        self.slowdown_dist = 0.55
        self.stop_dist = 0.22

        # Obstacle avoidance (front only) 
        self.rrt_enabled = True
        self.rrt_trigger_dist = 0.90
        self.rrt_clearance = 0.35
        self.backup_steps = 18
        self.backup_v = -0.08

        # RRT*
        self.rrt_max_iters = 2500
        self.rrt_step_size = 0.35
        self.rrt_neighbor_radius = 0.7
        self.rrt_goal_radius = 0.35
        self.rrt_goal_sample_rate = 0.10
        self.rrt_collision_step = 0.03

        self._front_block_count = 0
        self._front_block_needed = 3

        # Ball behavior 
        self.ball_radius_m = 0.15
        self.ball_standoff_m = 0.45
        self.ball_reach_tol = 0.25
        self.ball_inspect_wait = 0.55

        # Waypoint scan spin
        self.spin_w = 0.60
        self.spin_margin_rad = math.radians(12.0)

        # LiDAR cluster filters
        self.ball_cluster_halfwin = 8
        self.ball_cluster_minpts = 6
        self.ball_cluster_r_eps = 0.12
        self.ball_dup_reject_m = 0.60

        self.require_camera_color_gate = True  # recommended to reduce false positives
        self.ball_r_est_min = 0.10             # meters (tune)
        self.ball_r_est_max = 0.22             # meters (tune)
        self.ball_static_clearance_m = 0.10    # meters away from static inflated obstacles

        # Stop condition
        self.stop_when_all_colors_found = False
        self.found_colors = set()
        self.logged_ball_centers = []
        self.pending_color_hint = None

        # WAIT, FOLLOW_ASTAR, BACKUP_RRT, FOLLOW_RRT, SCAN_SPIN, BALL_GOTO, BALL_FACE, BALL_READ
        self.state = "WAIT"
        self.astar_path = Path()
        self.astar_progress_idx = 0
        self.rrt_path = Path()
        self.rrt_progress_idx = 0
        self.backup_remaining = 0

        # Spin bookkeeping
        self.spin_last_yaw = 0.0
        self.spin_accum = 0.0
        self.spin_active = False

        # Ball bookkeeping
        self.ball_center = None   # (cx, cy)
        self.ball_goal = None     # (gx, gy)
        self.ball_read_start = None

        self.map_loaded = False
        self.have_pose = False
        self.ttbot_pose = PoseStamped()
        self.scan_msg = None

        self.bridge = CvBridge()
        self.last_frame = None

        # Map arrays
        self.map_occ = None
        self.base_inflated = None
        self.inflated = None
        self.dist_to_occ = None
        self.dist_to_static = None  # NEW: distance to static inflated occupancy

        self.res = 0.05
        self.ox = 0.0
        self.oy = 0.0
        self.W = 0
        self.H = 0

        self.graph = None
        self.astar = None

        # subscribers
        self.create_subscription(PoseWithCovarianceStamped, "amcl_pose", self._pose_cb, 10)

        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(OccupancyGrid, "map", self._map_cb, map_qos)
        self.create_subscription(LaserScan, "scan", self._scan_cb, 10)
        self.create_subscription(RosImage, "/camera/image_raw", self._img_cb, 10)

        # publishers
        self.path_pub = self.create_publisher(Path, "global_plan", 10)
        self.cmd_vel_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.wp_marker_pub = self.create_publisher(MarkerArray, "/task3/waypoints", 1)
        self.goal_marker_pub = self.create_publisher(Marker, "/task3/current_goal", 1)

        # Required outputs
        self.red_pos_pub = self.create_publisher(Point, "/red_pos", 10)
        self.green_pos_pub = self.create_publisher(Point, "/green_pos", 10)
        self.blue_pos_pub = self.create_publisher(Point, "/blue_pos", 10)
        self.ball_log_pub = self.create_publisher(String, "/task3/ball_log", 10)

        # timers
        self.ctrl_timer = self.create_timer(0.1, self._control_loop)
        self.marker_timer = self.create_timer(1.0, self._publish_markers)

        self.get_logger().info("[READY] Waypoints + 360 scan + strict ball gating enabled.")

    # callbacks
    def _pose_cb(self, msg: PoseWithCovarianceStamped):
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose = msg.pose.pose
        self.ttbot_pose = ps
        self.have_pose = True

    def _scan_cb(self, msg: LaserScan):
        self.scan_msg = msg

    def _img_cb(self, msg: RosImage):
        try:
            self.last_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return

    def _map_cb(self, msg: OccupancyGrid):
        self.res = msg.info.resolution
        self.ox = msg.info.origin.position.x
        self.oy = msg.info.origin.position.y
        self.W = msg.info.width
        self.H = msg.info.height

        occ = np.array(msg.data, dtype=np.int16).reshape(self.H, self.W)
        if self.treat_unknown_as_occupied:
            occ[occ < 0] = 100
        self.map_occ = occ

        self.base_inflated = self._inflate_from_occ(self.map_occ, self.extra_inflation)
        self.inflated = self.base_inflated.copy()

        # NEW: distance field relative to static inflated obstacles
        self.dist_to_static = self._distance_to_occupied(self.base_inflated)

        self.dist_to_occ = self._distance_to_occupied(self.base_inflated)
        self.graph = self._build_graph_from_inflated(self.inflated, self.dist_to_occ)
        self.astar = AStar(self.graph)

        self.map_loaded = True
        self.get_logger().info(f"[MAP] Loaded {self.W}x{self.H} res={self.res:.3f}")
        self.get_logger().info(f"[GRAPH] nodes={len(self.graph.nodes)}")
        self._publish_markers()

    def _inflate_from_occ(self, occ_grid: np.ndarray, extra_inflation_m: float):
        H, W = occ_grid.shape
        base = np.where(occ_grid >= 50, 100, 0).astype(np.uint8)

        r_cells = int(math.ceil((self.robot_radius + float(extra_inflation_m)) / self.res))
        if r_cells <= 0:
            return base

        yy, xx = np.ogrid[-r_cells:r_cells + 1, -r_cells:r_cells + 1]
        disk = (xx * xx + yy * yy) <= (r_cells * r_cells)

        dil = base.copy()
        for dy in range(-r_cells, r_cells + 1):
            for dx in range(-r_cells, r_cells + 1):
                if not disk[dy + r_cells, dx + r_cells]:
                    continue
                src_y0 = max(0, -dy)
                src_y1 = H - max(0, dy)
                src_x0 = max(0, -dx)
                src_x1 = W - max(0, dx)

                dst_y0 = max(0, dy)
                dst_y1 = H - max(0, -dy)
                dst_x0 = max(0, dx)
                dst_x1 = W - max(0, -dx)

                dst = dil[dst_y0:dst_y1, dst_x0:dst_x1]
                src = base[src_y0:src_y1, src_x0:src_x1]
                np.maximum(dst, src, out=dst)
        return dil

    def _distance_to_occupied(self, inflated: np.ndarray):
        H, W = inflated.shape
        dist = np.full((H, W), -1, dtype=np.int16)
        q = deque()

        occ_cells = np.argwhere(inflated >= 50)
        for r, c in occ_cells:
            dist[r, c] = 0
            q.append((r, c))

        nbr4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while q:
            r, c = q.popleft()
            nd = dist[r, c] + 1
            for dr, dc in nbr4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and dist[rr, cc] < 0:
                    dist[rr, cc] = nd
                    q.append((rr, cc))

        dist[dist < 0] = 32767
        return dist

    def _build_graph_from_inflated(self, inflated: np.ndarray, dist_to_occ: np.ndarray):
        H, W = inflated.shape
        g = GridGraph()

        for r in range(H):
            for c in range(W):
                if inflated[r, c] == 0:
                    g.nodes[f"{r},{c}"] = GraphNode(f"{r},{c}")

        nbrs = [
            (-1,  0, 1.0), ( 1,  0, 1.0), ( 0, -1, 1.0), ( 0,  1, 1.0),
            (-1, -1, math.sqrt(2.0)), (-1,  1, math.sqrt(2.0)),
            ( 1, -1, math.sqrt(2.0)), ( 1,  1, math.sqrt(2.0)),
        ]

        for r in range(H):
            for c in range(W):
                if inflated[r, c] != 0:
                    continue
                parent = g.nodes.get(f"{r},{c}")
                if parent is None:
                    continue

                neighbors, weights = [], []
                for dr, dc, w in nbrs:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if inflated[nr, nc] != 0:
                        continue

                    # NO diagonal corner cutting
                    if dr != 0 and dc != 0:
                        if inflated[r + dr, c] != 0 or inflated[r, c + dc] != 0:
                            continue

                    cost = float(w)
                    if self.use_soft_clearance:
                        d_m = float(dist_to_occ[nr, nc]) * self.res
                        if d_m < self.soft_clearance_m:
                            penalty = self.soft_clearance_gain * (self.soft_clearance_m - d_m) / self.soft_clearance_m
                            cost *= (1.0 + penalty)

                    neighbors.append(g.nodes[f"{nr},{nc}"])
                    weights.append(cost)

                if neighbors:
                    parent.add_edges(neighbors, weights)
        return g

    def _in_map_world(self, x, y):
        return (self.ox <= x < self.ox + self.W * self.res) and (self.oy <= y < self.oy + self.H * self.res)

    def world_to_grid(self, x, y):
        col = int((x - self.ox) / self.res)
        row = int((y - self.oy) / self.res)
        col = max(0, min(self.W - 1, col))
        row = max(0, min(self.H - 1, row))
        return col, row

    def grid_to_world(self, col, row):
        x = self.ox + (col + 0.5) * self.res
        y = self.oy + (row + 0.5) * self.res
        return x, y

    def _nearest_free(self, col, row):
        H, W = self.H, self.W
        if 0 <= row < H and 0 <= col < W and self.inflated[row, col] == 0:
            return col, row

        visited = np.zeros((H, W), dtype=np.uint8)
        q = deque()
        c0 = max(0, min(W - 1, col))
        r0 = max(0, min(H - 1, row))
        q.append((c0, r0))
        visited[r0, c0] = 1

        nbr8 = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        while q:
            c, r = q.popleft()
            if self.inflated[r, c] == 0:
                return c, r
            for dc, dr in nbr8:
                nc, nr = c + dc, r + dr
                if 0 <= nc < W and 0 <= nr < H and not visited[nr, nc]:
                    visited[nr, nc] = 1
                    q.append((nc, nr))
        return c0, r0

    def _plan_astar_once(self, gx, gy):
        if not self.map_loaded or not self.have_pose:
            return False
        if not self._in_map_world(gx, gy):
            self.get_logger().warn("[A*] Goal out of bounds.")
            return False

        sx, sy = self.ttbot_pose.pose.position.x, self.ttbot_pose.pose.position.y
        sc, sr = self.world_to_grid(sx, sy)
        gc, gr = self.world_to_grid(gx, gy)

        sc, sr = self._nearest_free(sc, sr)
        gc, gr = self._nearest_free(gc, gr)

        s_name = f"{sr},{sc}"
        g_name = f"{gr},{gc}"

        if s_name not in self.graph.nodes or g_name not in self.graph.nodes:
            self.get_logger().warn("[A*] Start/goal not in free graph.")
            return False

        ok = self.astar.plan(s_name, g_name)
        names = self.astar.reconstruct(s_name, g_name) if ok else []
        if not names:
            self.get_logger().warn("[A*] No path.")
            return False

        path = Path()
        path.header.frame_id = "map"
        for nm in names:
            r, c = map(int, nm.split(","))
            wx, wy = self.grid_to_world(c, r)
            ps = PoseStamped()
            ps.header.frame_id = "map"
            ps.pose.position.x = float(wx)
            ps.pose.position.y = float(wy)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)

        self.astar_path = path
        self.astar_progress_idx = 0
        self.path_pub.publish(self.astar_path)
        return True

    # Pure pursuit 
    def _closest_index_windowed(self, path: Path, x, y, prev_idx: int):
        n = len(path.poses)
        if n == 0:
            return 0
        start = max(0, prev_idx - 25)
        end = min(n, prev_idx + 80) if prev_idx > 0 else n

        best_i, best_d2 = start, float("inf")
        for i in range(start, end):
            px = path.poses[i].pose.position.x
            py = path.poses[i].pose.position.y
            d2 = (px - x) ** 2 + (py - y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        return best_i

    def _lookahead_index(self, path: Path, from_idx: int, L: float):
        n = len(path.poses)
        if n == 0:
            return 0
        if from_idx >= n - 1:
            return n - 1

        accum = 0.0
        last_x = path.poses[from_idx].pose.position.x
        last_y = path.poses[from_idx].pose.position.y
        for i in range(from_idx + 1, n):
            x = path.poses[i].pose.position.x
            y = path.poses[i].pose.position.y
            accum += math.hypot(x - last_x, y - last_y)
            last_x, last_y = x, y
            if accum >= L:
                return i
        return n - 1

    def _pure_pursuit_cmd(self, path: Path, progress_idx: int, goal_x: float, goal_y: float):
        if not path.poses:
            return 0.0, 0.0, progress_idx

        rx = self.ttbot_pose.pose.position.x
        ry = self.ttbot_pose.pose.position.y
        yaw = get_yaw_from_quaternion(self.ttbot_pose.pose.orientation)

        ci = self._closest_index_windowed(path, rx, ry, progress_idx)
        ti = self._lookahead_index(path, ci, self.lookahead)

        tx = path.poses[ti].pose.position.x
        ty = path.poses[ti].pose.position.y

        dx = tx - rx
        dy = ty - ry
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        # world -> robot frame
        x_r = cos_y * dx + sin_y * dy
        y_r = -sin_y * dx + cos_y * dy

        if x_r < 0.05:
            w = max(-self.max_ang, min(self.max_ang, 0.9 * math.copysign(1.0, y_r)))
            return 0.0, w, ci

        L = max(self.lookahead, 1e-3)
        curvature = 2.0 * y_r / (L * L)

        dist_goal = math.hypot(goal_x - rx, goal_y - ry)
        goal_scale = min(1.0, max(0.15, dist_goal / 0.9))

        curv_scale = 1.0 / (1.0 + self.curvature_speed_k * abs(curvature))
        curv_scale = max(self.min_lin_scale, min(1.0, curv_scale))

        v = self.max_lin * goal_scale * curv_scale
        w = v * curvature
        w = max(-self.max_ang, min(self.max_ang, w))
        return v, w, ci

    # LiDAR helpers 
    def _front_min_distance(self, deg_min=-40, deg_max=40):
        if self.scan_msg is None:
            return float("inf")

        ranges = np.array(self.scan_msg.ranges, dtype=np.float32)
        angle_min = float(self.scan_msg.angle_min)
        angle_inc = float(self.scan_msg.angle_increment)
        n = len(ranges)
        angles = angle_min + angle_inc * np.arange(n, dtype=np.float32)

        a0 = math.radians(deg_min)
        a1 = math.radians(deg_max)
        mask = (angles >= a0) & (angles <= a1)
        sector = ranges[mask]
        sector = sector[np.isfinite(sector)]
        if sector.size == 0:
            return float("inf")
        return float(np.min(sector))

    def _apply_scan_slowdown(self, v_cmd, front_min):
        if not math.isfinite(front_min):
            return v_cmd
        if front_min <= self.stop_dist:
            return 0.0
        if front_min < self.slowdown_dist:
            scale = (front_min - self.stop_dist) / (self.slowdown_dist - self.stop_dist)
            scale = max(0.0, min(1.0, scale))
            return v_cmd * scale
        return v_cmd

    def _point_is_static_occupied(self, x, y) -> bool:
        if self.map_occ is None or (not self._in_map_world(x, y)):
            return True
        c, r = self.world_to_grid(x, y)
        return self.map_occ[r, c] >= 50

    def _point_is_inflated_occupied(self, x, y) -> bool:
        if self.inflated is None or (not self._in_map_world(x, y)):
            return True
        c, r = self.world_to_grid(x, y)
        return self.inflated[r, c] >= 50

    # Dynamic painting (existing mechanism) 
    def mark_dynamic_obstacle_world(self, x, y, radius=None):
        if self.inflated is None:
            return
        if radius is None:
            radius = self.robot_radius + self.rrt_clearance

        c_center, r_center = self.world_to_grid(x, y)
        r_cells = int(math.ceil(float(radius) / self.res))

        for dr in range(-r_cells, r_cells + 1):
            for dc in range(-r_cells, r_cells + 1):
                rr = r_center + dr
                cc = c_center + dc
                if 0 <= rr < self.H and 0 <= cc < self.W:
                    if dr * dr + dc * dc <= r_cells * r_cells:
                        self.inflated[rr, cc] = 100

        self.dist_to_occ = self._distance_to_occupied(self.inflated)
        self.graph = self._build_graph_from_inflated(self.inflated, self.dist_to_occ)
        self.astar = AStar(self.graph)

    #  RRT* 
    def _is_free_world(self, x, y):
        if not self._in_map_world(x, y):
            return False
        c, r = self.world_to_grid(x, y)
        return self.inflated[r, c] == 0

    def _collision_free_segment(self, x1, y1, x2, y2):
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist < 1e-6:
            return self._is_free_world(x1, y1)
        steps = int(dist / self.rrt_collision_step) + 1
        for i in range(steps + 1):
            t = i / max(steps, 1)
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            if not self._is_free_world(x, y):
                return False
        return True

    def _rrt_star_plan(self, start_xy, goal_xy):
        sx, sy = start_xy
        gx, gy = goal_xy

        if not self._is_free_world(sx, sy) or not self._is_free_world(gx, gy):
            return [], None

        x_min = self.ox
        x_max = self.ox + self.W * self.res
        y_min = self.oy
        y_max = self.oy + self.H * self.res

        nodes = [RRTStarNode(sx, sy, parent=None, cost=0.0)]
        goal_idx = None
        best_cost = float("inf")
        found_once = False

        for it in range(self.rrt_max_iters):
            if np.random.rand() < self.rrt_goal_sample_rate:
                x_rand, y_rand = gx, gy
            else:
                x_rand = float(np.random.uniform(x_min, x_max))
                y_rand = float(np.random.uniform(y_min, y_max))

            if not self._is_free_world(x_rand, y_rand):
                continue

            idx_near = min(range(len(nodes)),
                           key=lambda i: (nodes[i].x - x_rand) ** 2 + (nodes[i].y - y_rand) ** 2)
            n_near = nodes[idx_near]

            theta = math.atan2(y_rand - n_near.y, x_rand - n_near.x)
            dist = math.hypot(x_rand - n_near.x, y_rand - n_near.y)
            step = min(self.rrt_step_size, dist)
            x_new = n_near.x + step * math.cos(theta)
            y_new = n_near.y + step * math.sin(theta)

            if not self._is_free_world(x_new, y_new):
                continue
            if not self._collision_free_segment(n_near.x, n_near.y, x_new, y_new):
                continue

            new_node = RRTStarNode(x_new, y_new, parent=idx_near, cost=n_near.cost + step)

            idx_neighbors = []
            for i, ni in enumerate(nodes):
                d = math.hypot(ni.x - x_new, ni.y - y_new)
                if d <= self.rrt_neighbor_radius:
                    idx_neighbors.append((i, d))

            for i_n, d in idx_neighbors:
                if i_n == idx_near:
                    continue
                ni = nodes[i_n]
                if not self._collision_free_segment(ni.x, ni.y, x_new, y_new):
                    continue
                new_cost = ni.cost + d
                if new_cost < new_node.cost:
                    new_node.cost = new_cost
                    new_node.parent = i_n

            nodes.append(new_node)
            new_idx = len(nodes) - 1

            for i_n, d in idx_neighbors:
                if i_n == new_idx:
                    continue
                ni = nodes[i_n]
                if not self._collision_free_segment(new_node.x, new_node.y, ni.x, ni.y):
                    continue
                alt_cost = new_node.cost + d
                if alt_cost < ni.cost:
                    ni.cost = alt_cost
                    ni.parent = new_idx

            d_goal = math.hypot(x_new - gx, y_new - gy)
            if d_goal <= self.rrt_goal_radius and self._collision_free_segment(x_new, y_new, gx, gy):
                goal_cost = new_node.cost + d_goal
                if goal_cost < best_cost:
                    goal_node = RRTStarNode(gx, gy, parent=new_idx, cost=goal_cost)
                    nodes.append(goal_node)
                    goal_idx = len(nodes) - 1
                    best_cost = goal_cost
                    found_once = True

            if found_once and it > self.rrt_max_iters * 0.6:
                break

        return nodes, goal_idx

    def _plan_rrt_once(self, gx, gy):
        if not self.map_loaded or not self.have_pose:
            return False
        if not self._in_map_world(gx, gy):
            return False

        sx, sy = self.ttbot_pose.pose.position.x, self.ttbot_pose.pose.position.y
        sc, sr = self.world_to_grid(sx, sy)
        gc, gr = self.world_to_grid(gx, gy)
        sc, sr = self._nearest_free(sc, sr)
        gc, gr = self._nearest_free(gc, gr)

        sx, sy = self.grid_to_world(sc, sr)
        gx, gy = self.grid_to_world(gc, gr)

        nodes, goal_idx = self._rrt_star_plan((sx, sy), (gx, gy))
        if goal_idx is None:
            return False

        pts = []
        idx = goal_idx
        while idx is not None:
            n = nodes[idx]
            pts.append((n.x, n.y))
            idx = n.parent
        pts.reverse()

        path = Path()
        path.header.frame_id = "map"
        for (wx, wy) in pts:
            ps = PoseStamped()
            ps.header.frame_id = "map"
            ps.pose.position.x = float(wx)
            ps.pose.position.y = float(wy)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)

        self.rrt_path = path
        self.rrt_progress_idx = 0
        self.path_pub.publish(self.rrt_path)
        return True

    def _publish_markers(self):
        if not self.map_loaded:
            return
        now = self.get_clock().now().to_msg()
        ma = MarkerArray()
        for i, (x, y) in enumerate(self.waypoints):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = now
            m.ns = "task3_waypoints"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = 0.10
            m.pose.orientation.w = 1.0
            m.scale.x = 0.18
            m.scale.y = 0.18
            m.scale.z = 0.18
            m.color.r = 0.2
            m.color.g = 1.0
            m.color.b = 0.2
            m.color.a = 0.85
            ma.markers.append(m)
        self.wp_marker_pub.publish(ma)

        if 0 <= self.wp_idx < len(self.waypoints):
            gx, gy = self.waypoints[self.wp_idx]
            g = Marker()
            g.header.frame_id = "map"
            g.header.stamp = now
            g.ns = "task3_current_goal"
            g.id = 0
            g.type = Marker.ARROW
            g.action = Marker.ADD
            g.pose.position.x = float(gx)
            g.pose.position.y = float(gy)
            g.pose.position.z = 0.15
            g.pose.orientation.w = 1.0
            g.scale.x = 0.45
            g.scale.y = 0.10
            g.scale.z = 0.10
            g.color.r = 1.0
            g.color.g = 0.6
            g.color.b = 0.0
            g.color.a = 0.95
            self.goal_marker_pub.publish(g)
    
    def move_ttbot(self, v, w):
        if not rclpy.ok():
            return
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        try:
            self.cmd_vel_pub.publish(cmd)
        except Exception:
            pass
 
    def _start_waypoint(self):
        if self.wp_idx >= len(self.waypoints):
            self.state = "WAIT"
            return
        gx, gy = self.waypoints[self.wp_idx]

        if self._plan_astar_once(gx, gy):
            self.state = "FOLLOW_ASTAR"
            return

        # Relax inflation a bit, then RRT once
        old_extra = self.extra_inflation
        for extra in [max(0.06, old_extra - 0.03), max(0.06, old_extra - 0.06)]:
            self.base_inflated = self._inflate_from_occ(self.map_occ, extra)
            self.inflated = self.base_inflated.copy()

            self.dist_to_static = self._distance_to_occupied(self.base_inflated)

            self.dist_to_occ = self._distance_to_occupied(self.inflated)
            self.graph = self._build_graph_from_inflated(self.inflated, self.dist_to_occ)
            self.astar = AStar(self.graph)
            if self._plan_astar_once(gx, gy):
                self.extra_inflation = extra
                self.state = "FOLLOW_ASTAR"
                return

        if self._plan_rrt_once(gx, gy):
            self.state = "FOLLOW_RRT"
            return

        self.get_logger().error("[PLAN] No A* and no RRT*. Stopping.")
        self.state = "WAIT"

    def _camera_color_hint_fast(self):
        """Quick color-blob gate during scan spin. Returns 'red'/'green'/'blue'/None."""
        if self.last_frame is None:
            return None

        frame = self.last_frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        H, W = hsv.shape[:2]

        palette = {
            "red": [
                (np.array([0, 140, 90], np.uint8), np.array([10, 255, 255], np.uint8)),
                (np.array([170, 140, 90], np.uint8), np.array([179, 255, 255], np.uint8)),
            ],
            "green": [
                (np.array([40, 120, 80], np.uint8), np.array([85, 255, 255], np.uint8)),
            ],
            "blue": [
                (np.array([105, 140, 70], np.uint8), np.array([125, 255, 255], np.uint8)),
            ],
        }

        for color, ranges in palette.items():
            if color in self.found_colors:
                continue

            mask = None
            for lo, hi in ranges:
                m = cv2.inRange(hsv, lo, hi)
                mask = m if mask is None else cv2.bitwise_or(mask, m)

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                A = float(cv2.contourArea(c))
                if A < 250.0:
                    continue
                peri = float(cv2.arcLength(c, True))
                if peri <= 1e-6:
                    continue
                circ = (4.0 * math.pi * A) / (peri * peri)
                if circ < 0.35:
                    continue

                M = cv2.moments(c)
                if abs(M.get("m00", 0.0)) < 1e-9:
                    continue
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])

                if abs(cx - 0.5 * W) > 0.40 * W or abs(cy - 0.5 * H) > 0.50 * H:
                    continue

                return color

        return None

    def _static_clearance_ok(self, wx: float, wy: float) -> bool:
        """Reject points too close to static inflated obstacles."""
        if self.dist_to_static is None or not self._in_map_world(wx, wy):
            return False
        c, r = self.world_to_grid(wx, wy)
        d_m = float(self.dist_to_static[r, c]) * self.res
        return d_m >= self.ball_static_clearance_m

    def _expand_cluster(self, ranges, k, r0, eps):
        """Expand contiguous cluster around index k where |r-r0|<eps."""
        n = len(ranges)
        L = k
        while L > 0 and math.isfinite(ranges[L - 1]) and abs(float(ranges[L - 1]) - r0) < eps:
            L -= 1
        R = k
        while R < n - 1 and math.isfinite(ranges[R + 1]) and abs(float(ranges[R + 1]) - r0) < eps:
            R += 1
        return L, R

    def _cluster_ball_radius_est_ok(self, r_surf, left_i, right_i, angle_inc, ranges):
        """
        Radius estimate from angular width of cluster.
        R ≈ r * sin(theta/2) / (1 - sin(theta/2)).
        Also require a mild "roundness" check: cluster ends are farther than center.
        """
        npts = (right_i - left_i + 1)
        if npts < self.ball_cluster_minpts:
            return False, None

        theta = max(1e-6, npts * float(angle_inc))
        s = math.sin(0.5 * theta)
        if s <= 1e-3 or s >= 0.95:
            return False, None

        R_est = float(r_surf) * s / max(1e-6, (1.0 - s))
        if not (self.ball_r_est_min <= R_est <= self.ball_r_est_max):
            return False, None

        # mild shape sanity: ends should be (usually) a bit farther than middle
        rL = float(ranges[left_i]) if math.isfinite(ranges[left_i]) else r_surf
        rR = float(ranges[right_i]) if math.isfinite(ranges[right_i]) else r_surf
        if (rL < r_surf - 0.05) or (rR < r_surf - 0.05):
            return False, None

        return True, R_est

    def _scan_for_ball_candidate(self):
        """Return (cx,cy) if a strict ball-like candidate exists; else None."""
        if self.scan_msg is None or (not self.have_pose):
            return None

        hint = self._camera_color_hint_fast()
        if self.require_camera_color_gate and hint is None:
            return None

        scan = self.scan_msg
        rx = self.ttbot_pose.pose.position.x
        ry = self.ttbot_pose.pose.position.y

        ranges = scan.ranges
        n = len(ranges)

        best = None  # (range, (cx,cy), hint)

        for k in range(n):
            r0 = ranges[k]
            if not math.isfinite(r0):
                continue
            r0 = float(r0)
            if r0 < scan.range_min or r0 > scan.range_max:
                continue

            # local continuity check first (fast reject)
            lo = max(0, k - self.ball_cluster_halfwin)
            hi = min(n - 1, k + self.ball_cluster_halfwin)
            cnt = 0
            for kk in range(lo, hi + 1):
                rr = ranges[kk]
                if not math.isfinite(rr):
                    continue
                rr = float(rr)
                if rr < scan.range_min or rr > scan.range_max:
                    continue
                if abs(rr - r0) <= self.ball_cluster_r_eps:
                    cnt += 1
            if cnt < self.ball_cluster_minpts:
                continue

            # grow contiguous cluster
            left, right = self._expand_cluster(ranges, k, r0, eps=self.ball_cluster_r_eps)
            ok, R_est = self._cluster_ball_radius_est_ok(r0, left, right, scan.angle_increment, ranges)
            if not ok:
                continue

            # surface hit -> world
            a = scan.angle_min + k * scan.angle_increment
            wx_surf = rx + r0 * math.cos(get_yaw_from_quaternion(self.ttbot_pose.pose.orientation) + a)
            wy_surf = ry + r0 * math.sin(get_yaw_from_quaternion(self.ttbot_pose.pose.orientation) + a)

            if not self._static_clearance_ok(wx_surf, wy_surf):
                continue

            # center estimate
            dx = wx_surf - rx
            dy = wy_surf - ry
            norm = math.hypot(dx, dy)
            if norm < 1e-6:
                continue
            ux = dx / norm
            uy = dy / norm

            center_d = r0 + float(R_est)
            cx = rx + ux * center_d
            cy = ry + uy * center_d

            if not self._in_map_world(cx, cy):
                continue

            # reject duplicates
            dup = False
            for (px, py) in self.logged_ball_centers:
                if math.hypot(cx - px, cy - py) < self.ball_dup_reject_m:
                    dup = True
                    break
            if dup:
                continue

            if best is None or r0 < best[0]:
                best = (r0, (cx, cy), hint)

        if best is None:
            return None

        self.pending_color_hint = best[2]
        return best[1]

    def _make_ball_standoff_goal(self, cx: float, cy: float):
        rx = self.ttbot_pose.pose.position.x
        ry = self.ttbot_pose.pose.position.y
        dx = cx - rx
        dy = cy - ry
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return None
        ux = dx / d
        uy = dy / d

        goal_d = max(0.10, d - max(0.05, self.ball_standoff_m))
        gx = rx + ux * goal_d
        gy = ry + uy * goal_d
        if not self._in_map_world(gx, gy):
            return None

        # also ensure stand-off goal is not too close to static walls
        if not self._static_clearance_ok(gx, gy):
            return None
        return (gx, gy)

    def _refine_ball_center_front_arc(self):
        """Refine center using front arc closest hit."""
        if self.scan_msg is None or not self.have_pose:
            return False

        scan = self.scan_msg
        rx = self.ttbot_pose.pose.position.x
        ry = self.ttbot_pose.pose.position.y
        yaw = get_yaw_from_quaternion(self.ttbot_pose.pose.orientation)

        best_r = math.inf
        best_surface = None
        max_a = 0.45

        for k, r in enumerate(scan.ranges):
            if not math.isfinite(r):
                continue
            r = float(r)
            if r < scan.range_min or r > scan.range_max:
                continue

            a = scan.angle_min + k * scan.angle_increment
            if abs(a) > max_a:
                continue

            wx = rx + r * math.cos(yaw + a)
            wy = ry + r * math.sin(yaw + a)

            if self._point_is_static_occupied(wx, wy):
                continue

            if r < best_r:
                best_r = r
                best_surface = (wx, wy)

        if best_surface is None or not math.isfinite(best_r):
            return False

        dx = best_surface[0] - rx
        dy = best_surface[1] - ry
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            return False
        ux = dx / norm
        uy = dy / norm

        center_d = best_r + self.ball_radius_m
        self.ball_center = (rx + ux * center_d, ry + uy * center_d)
        return True

    def _choose_ball_color(self):
        """HSV + contour checks; returns 'red'/'green'/'blue'/None."""
        if self.last_frame is None:
            return None

        frame = self.last_frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        H, W = hsv.shape[:2]

        palette = {
            "red": [
                (np.array([0, 140, 90], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8)),
                (np.array([170, 140, 90], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8)),
            ],
            "green": [
                (np.array([40, 120, 80], dtype=np.uint8), np.array([85, 255, 255], dtype=np.uint8)),
            ],
            "blue": [
                (np.array([105, 140, 70], dtype=np.uint8), np.array([125, 255, 255], dtype=np.uint8)),
            ],
        }

        def dominance_ok(label: str, mean_bgr):
            b, g, r = float(mean_bgr[0]), float(mean_bgr[1]), float(mean_bgr[2])
            k = 1.20
            if label == "red":
                return (r > k * g) and (r > k * b)
            if label == "green":
                return (g > k * r) and (g > k * b)
            if label == "blue":
                return (b > k * r) and (b > k * g)
            return False

        best_label = None
        best_score = 0.0

        for label, ranges in palette.items():
            if label in self.found_colors:
                continue

            mask = None
            for lo, hi in ranges:
                m = cv2.inRange(hsv, lo, hi)
                mask = m if mask is None else cv2.bitwise_or(mask, m)

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                A = float(cv2.contourArea(c))
                if A < 400.0:
                    continue
                peri = float(cv2.arcLength(c, True))
                if peri <= 1e-6:
                    continue
                circ = (4.0 * math.pi * A) / (peri * peri)
                if circ < 0.45:
                    continue

                M = cv2.moments(c)
                if abs(M.get("m00", 0.0)) < 1e-9:
                    continue
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])

                if abs(cx - 0.5 * W) > 0.35 * W or abs(cy - 0.5 * H) > 0.45 * H:
                    continue

                cmask = np.zeros((H, W), dtype=np.uint8)
                cv2.drawContours(cmask, [c], -1, 255, thickness=-1)
                mean_bgr = cv2.mean(frame, mask=cmask)[:3]
                if not dominance_ok(label, mean_bgr):
                    continue

                score = A * circ
                if score > best_score:
                    best_score = score
                    best_label = label

        return best_label

    def _publish_color_position(self, color: str, cx: float, cy: float):
        p = Point()
        p.x = float(cx)
        p.y = float(cy)
        p.z = 0.0

        log = String()
        log.data = f"{color} {cx:.3f} {cy:.3f}"
        self.ball_log_pub.publish(log)

        if color == "red":
            self.red_pos_pub.publish(p)
        elif color == "green":
            self.green_pos_pub.publish(p)
        elif color == "blue":
            self.blue_pos_pub.publish(p)

    def _record_object_on_costmap(self, cx: float, cy: float, is_ball: bool):
        """Paint region to prevent repeated triggers."""
        if is_ball:
            rad = self.ball_radius_m + 0.15 + 0.08
        else:
            rad = 0.55  # smaller than 0.65 to avoid blocking corridors too aggressively
        self.mark_dynamic_obstacle_world(cx, cy, radius=rad)
        self.logged_ball_centers.append((cx, cy))

    def _begin_spin(self):
        yaw = get_yaw_from_quaternion(self.ttbot_pose.pose.orientation)
        self.spin_last_yaw = yaw
        self.spin_accum = 0.0
        self.spin_active = True
        self.state = "SCAN_SPIN"
        self.move_ttbot(0.0, 0.0)

    def _spin_step(self):
        yaw = get_yaw_from_quaternion(self.ttbot_pose.pose.orientation)
        dyaw = wrap_to_pi(yaw - self.spin_last_yaw)
        self.spin_accum += abs(dyaw)
        self.spin_last_yaw = yaw

        # during spin: look for a ball
        if (not self.stop_when_all_colors_found) or (len(self.found_colors) < 3):
            cand = self._scan_for_ball_candidate()
            if cand is not None:
                cx, cy = cand
                goal = self._make_ball_standoff_goal(cx, cy)
                if goal is not None:
                    gx, gy = goal
                    if self._plan_astar_once(gx, gy):
                        self.ball_center = (cx, cy)
                        self.ball_goal = (gx, gy)
                        self.state = "BALL_GOTO"
                        self.spin_active = False
                        self.move_ttbot(0.0, 0.0)
                        self.get_logger().info(f"[BALL] Candidate -> go stand-off ({gx:.2f},{gy:.2f})")
                        return

        if self.spin_accum >= (2.0 * math.pi - self.spin_margin_rad):
            self.move_ttbot(0.0, 0.0)
            self.spin_active = False
            self.wp_idx += 1

            self.astar_path = Path()
            self.rrt_path = Path()
            self.astar_progress_idx = 0
            self.rrt_progress_idx = 0
            self._front_block_count = 0
            self.backup_remaining = 0
            self.state = "WAIT"
            return

        self.move_ttbot(0.0, sat(self.spin_w, -self.max_ang, self.max_ang))

    def _estimate_front_obstacle_world(self, deg_min=-30, deg_max=30):
        if self.scan_msg is None or (not self.have_pose):
            return None

        ranges = np.array(self.scan_msg.ranges, dtype=np.float32)
        angle_min = float(self.scan_msg.angle_min)
        angle_inc = float(self.scan_msg.angle_increment)
        n = len(ranges)
        angles = angle_min + angle_inc * np.arange(n, dtype=np.float32)

        a0 = math.radians(deg_min)
        a1 = math.radians(deg_max)
        mask = (angles >= a0) & (angles <= a1)
        rr = ranges[mask]
        aa = angles[mask]

        finite = np.isfinite(rr)
        if not finite.any():
            return None

        rr2 = rr[finite]
        aa2 = aa[finite]
        idx = int(np.argmin(rr2))
        r = float(rr2[idx])
        ang = float(aa2[idx])

        if not math.isfinite(r) or r <= 0.05:
            return None
        if r < float(self.scan_msg.range_min) or r > float(self.scan_msg.range_max):
            return None

        rx = self.ttbot_pose.pose.position.x
        ry = self.ttbot_pose.pose.position.y
        yaw = get_yaw_from_quaternion(self.ttbot_pose.pose.orientation)

        theta = yaw + ang
        xw = rx + r * math.cos(theta)
        yw = ry + r * math.sin(theta)
        return xw, yw, r

    # main control
    def _control_loop(self):
        if not self.map_loaded or not self.have_pose:
            return

        if self.stop_when_all_colors_found and {"red", "green", "blue"}.issubset(self.found_colors):
            self.move_ttbot(0.0, 0.0)
            return

        if self.wp_idx >= len(self.waypoints):
            self.move_ttbot(0.0, 0.0)
            return

        # spinning?
        if self.state == "SCAN_SPIN" and self.spin_active:
            self._spin_step()
            return

        gx_wp, gy_wp = self.waypoints[self.wp_idx]
        rx = self.ttbot_pose.pose.position.x
        ry = self.ttbot_pose.pose.position.y

    
        if self.state == "BALL_GOTO":
            if self.ball_goal is None:
                self._begin_spin()
                return

            gx, gy = self.ball_goal
            if math.hypot(gx - rx, gy - ry) <= self.ball_reach_tol:
                self.move_ttbot(0.0, 0.0)
                self.state = "BALL_FACE"
                return

            front_min = self._front_min_distance(-40, 40)
            v, w, new_idx = self._pure_pursuit_cmd(self.astar_path, self.astar_progress_idx, gx, gy)
            self.astar_progress_idx = new_idx
            v = self._apply_scan_slowdown(v, front_min)
            self.move_ttbot(v, w)
            return

        if self.state == "BALL_FACE":
            if self.ball_center is None:
                self._begin_spin()
                return

            bx, by = self.ball_center
            yaw = get_yaw_from_quaternion(self.ttbot_pose.pose.orientation)
            desired = math.atan2(by - ry, bx - rx)
            err = wrap_to_pi(desired - yaw)

            if abs(err) > math.radians(10.0):
                w = sat(2.5 * err, -self.max_ang, self.max_ang)
                self.move_ttbot(0.0, w)
                return

            self.move_ttbot(0.0, 0.0)
            self.ball_read_start = self.get_clock().now().nanoseconds * 1e-9
            self.state = "BALL_READ"
            return

        if self.state == "BALL_READ":
            t_now = self.get_clock().now().nanoseconds * 1e-9
            if self.ball_read_start is None:
                self.ball_read_start = t_now

            if (t_now - self.ball_read_start) < self.ball_inspect_wait:
                self.move_ttbot(0.0, 0.0)
                return

            self._refine_ball_center_front_arc()
            if self.ball_center is None:
                self._begin_spin()
                return

            cx, cy = self.ball_center
            color = self._choose_ball_color()

            if color is not None:
                self.found_colors.add(color)
                self._publish_color_position(color, cx, cy)
                self._record_object_on_costmap(cx, cy, is_ball=True)
                self.get_logger().info(f"[BALL] {color} @ ({cx:.2f},{cy:.2f}) published.")
            else:
                self._record_object_on_costmap(cx, cy, is_ball=False)
                self.get_logger().info(f"[BALL] Non-color object @ ({cx:.2f},{cy:.2f}) painted.")

            # clear and resume spin
            self.ball_center = None
            self.ball_goal = None
            self.ball_read_start = None
            self.astar_path = Path()
            self.astar_progress_idx = 0
            self._begin_spin()
            return

        
        if math.hypot(gx_wp - rx, gy_wp - ry) <= self.goal_tol and self.state != "BACKUP_RRT":
            self.move_ttbot(0.0, 0.0)
            self.get_logger().info(f"[WP] Reached {self.wp_idx + 1}/{len(self.waypoints)} -> 360 scan")

            self.astar_path = Path()
            self.rrt_path = Path()
            self.astar_progress_idx = 0
            self.rrt_progress_idx = 0
            self._front_block_count = 0
            self.backup_remaining = 0

            self._begin_spin()
            return

        if self.state == "WAIT":
            self._start_waypoint()
            if self.state == "WAIT":
                self.move_ttbot(0.0, 0.0)
            return

        front_min = self._front_min_distance(-40, 40)

        # Trigger avoidance only while following A*
        if self.state == "FOLLOW_ASTAR" and self.rrt_enabled:
            self._front_block_count = self._front_block_count + 1 if front_min < self.rrt_trigger_dist else 0

            if self._front_block_count >= self._front_block_needed:
                est = self._estimate_front_obstacle_world(-40, 40)
                if est is not None:
                    ox, oy, d = est
                    if (not self._point_is_static_occupied(ox, oy)) and (not self._point_is_inflated_occupied(ox, oy)):
                        self.get_logger().info(
                            f"[RRT-TRIGGER] Obstacle at ({ox:.2f},{oy:.2f}) d={d:.2f} -> paint + backup + RRT*"
                        )
                        self.mark_dynamic_obstacle_world(ox, oy)
                        self.backup_remaining = self.backup_steps
                        self.state = "BACKUP_RRT"
                        self.move_ttbot(0.0, 0.0)
                        self._front_block_count = 0
                        return
                self._front_block_count = 0

        if self.state == "BACKUP_RRT":
            if self.backup_remaining > 0:
                self.backup_remaining -= 1
                self.move_ttbot(self.backup_v, 0.0)
                return

            if self._plan_rrt_once(gx_wp, gy_wp):
                self.state = "FOLLOW_RRT"
                return

            self.get_logger().warn("[RRT*] Failed -> fallback to A* (replan once).")
            if self._plan_astar_once(gx_wp, gy_wp):
                self.state = "FOLLOW_ASTAR"
            else:
                self.state = "WAIT"
            return

        if self.state == "FOLLOW_ASTAR":
            v, w, new_idx = self._pure_pursuit_cmd(self.astar_path, self.astar_progress_idx, gx_wp, gy_wp)
            self.astar_progress_idx = new_idx
            v = self._apply_scan_slowdown(v, front_min)
            self.move_ttbot(v, w)
            return

        if self.state == "FOLLOW_RRT":
            v, w, new_idx = self._pure_pursuit_cmd(self.rrt_path, self.rrt_progress_idx, gx_wp, gy_wp)
            self.rrt_progress_idx = new_idx
            v = self._apply_scan_slowdown(v, front_min)
            self.move_ttbot(v, w)
            return


def main(args=None):
    rclpy.init(args=args)
    node = Task3Navigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                node.move_ttbot(0.0, 0.0)
        except Exception:
            pass
        try:
            node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()