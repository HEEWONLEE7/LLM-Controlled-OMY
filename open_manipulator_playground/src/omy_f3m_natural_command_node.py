# natural_command_node.py — LLM-only JSON parser integration
# 2025-08-12 — add continuous rotate with "keep" + stop (comments in English)

import re
import json
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import GripperCommand
from control_msgs.msg import GripperCommand as GripperCommandMsg
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK
from tf2_ros import Buffer, TransformListener

import threading

try:
    from omy_f3m_llm_parser import parse_to_json
    HAS_LLM = True
except Exception as e:
    HAS_LLM = False
    print(f"[LLM] import failed: {e}")

BASE_HEIGHT = 0.1715
L1 = 0.1715
L2 = 0.247
L3 = 0.2195
L4 = 0.1155
L5 = 0.113
L6 = 0.1155
JOINT2_OFFSET = 0.1215
JOINT4_OFFSET = 0.1215
JOINT5_OFFSET = 0.113
L_GRIPPER = 0.1155
KEEP_ROTATE_SPEED_DEG_S = 10.0
KEEP_TIMER_HZ = 20.0
KEEP_DT = 1.0 / KEEP_TIMER_HZ


class NaturalCommandNode(Node):
    def __init__(self):
        super().__init__('natural_command_node')

        self.arm_pub = self.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.gripper_client = ActionClient(self, GripperCommand, '/gripper_controller/gripper_cmd')
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('⏳ Waiting for /compute_ik service...')

        self.current_joint1_pos = 0.0
        self.current_joint2_pos = 0.0
        self.current_joint3_pos = 0.0
        self.current_joint4_pos = 0.0
        self.current_joint5_pos = 0.0
        self.current_joint6_pos = 0.0
        self.current_z = L2

        self.declare_parameter('use_llm', True)
        self.use_llm = bool(self.get_parameter('use_llm').value) and HAS_LLM
        if not HAS_LLM:
            self.get_logger().warn("⚠️ LLM parser not available; commands will be ignored (no legacy regex).")
        else:
            self.get_logger().info(f"✅ LLM parser ready (use_llm={self.use_llm})")

        self._keep_timer = None
        self._keep_dir = None
        self._keep_w_rad_s = math.radians(KEEP_ROTATE_SPEED_DEG_S)

    def get_current_ee_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                "world", "end_effector_link", rclpy.time.Time()
            )
            pose = PoseStamped()
            pose.header.frame_id = "world"
            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z
            pose.pose.orientation = transform.transform.rotation
            return pose
        except Exception as e:
            self.get_logger().error(f"❌ Failed to get current EE pose: {e}")
            return None

    def parse_command_with_llm(self, text: str):
        if not self.use_llm:
            return None
        try:
            data = parse_to_json(text)
            if not isinstance(data, dict):
                return None
            for k in ["action", "direction", "value", "unit", "xyz"]:
                if k not in data:
                    return None
            if data["action"] is None:
                return None
            return data
        except Exception as e:
            self.get_logger().error(f"[LLM] parse failed: {e}")
            return None
        
    def parse_command_with_llm(self, text: str):
        if not self.use_llm:
            return None
        try:
            data = parse_to_json(text)
            if not isinstance(data, dict):
                return None
            for k in ["action", "direction", "value", "unit", "xyz"]:
                if k not in data:
                    return None
            if data["action"] is None:
                return None
            return data
        except Exception as e:
            self.get_logger().error(f"[LLM] parse failed: {e}")
            return None

    def process_command(self, cmd):
        try:
            action = cmd.get("action")

            if action == "stop":
                self._stop_keep()
                self.get_logger().info("⛔ Stopped continuous rotation.")
                return

            if action == "move_xyz":
                x, y, z = cmd["xyz"]
                # ✅ orientation 전달 가능하도록 변경
                self.send_ik_request(x, y, z, None)
                return

            if action == "initialize":
                self.reset_pose()
                return

            if action == "gripper":
                pos_map = {"open": 0, "close": 1, "reset": 0}
                pos = pos_map.get(str(cmd.get("direction") or "").lower(), None)
                if pos is None:
                    self.get_logger().warn(f"⚠️ Unknown gripper direction: {cmd.get('direction')}")
                    return
                goal = GripperCommand.Goal()
                goal.command = GripperCommandMsg()
                goal.command.position = float(pos)
                goal.command.max_effort = 10.0
                self.gripper_client.wait_for_server()
                self.gripper_client.send_goal_async(goal)
                self.get_logger().info(f"🦾 Gripper '{cmd['direction']}' executed")
                return

            if action in ("rotate", "move"):
                direction = cmd.get("direction")
                value = cmd.get("value")
                unit = cmd.get("unit")

                if (action == "rotate") and isinstance(value, str) and value.strip().lower() == "keep":
                    self._start_keep(direction)
                    return

                if value is None or unit is None or direction is None:
                    self.get_logger().warn(f"⚠️ Incomplete fields for {action}: {cmd}")
                    return

                traj = JointTrajectory()
                traj.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
                point = JointTrajectoryPoint()
                radius = L3 * math.cos(self.current_joint3_pos) + L_GRIPPER
                delta = self.get_delta(value, unit, radius)

                if action == "rotate":
                    if direction == "left":
                        self.current_joint1_pos += delta
                    elif direction == "right":
                        self.current_joint1_pos -= delta
                    else:
                        self.get_logger().warn(f"⚠️ Unsupported rotate direction: {direction}")
                        return

                if action == "move":
                    # ✅ 현재 EE pose 조회
                    curr_pose = self.get_current_ee_pose()
                    if curr_pose is None:
                        return

                    goal_pose = PoseStamped()
                    goal_pose.header.frame_id = "world"
                    goal_pose.pose.position.x = curr_pose.pose.position.x
                    goal_pose.pose.position.y = curr_pose.pose.position.y
                    goal_pose.pose.position.z = curr_pose.pose.position.z
                    goal_pose.pose.orientation = curr_pose.pose.orientation  # 방향 유지

                    step = value / 100.0 if unit.lower() == "cm" else float(value)

                    if direction == "up":
                        goal_pose.pose.position.z += step
                    elif direction == "down":
                        goal_pose.pose.position.z -= step
                    elif direction == "forward":
                        goal_pose.pose.position.x += step
                    elif direction == "backward":
                        goal_pose.pose.position.x -= step
                    elif direction == "left":
                        goal_pose.pose.position.y += step
                    elif direction == "right":
                        goal_pose.pose.position.y -= step
                    else:
                        self.get_logger().warn(f"⚠️ Unknown move direction: {direction}")
                        return

                    self.send_ik_request(
                        goal_pose.pose.position.x,
                        goal_pose.pose.position.y,
                        goal_pose.pose.position.z,
                        goal_pose.pose.orientation
                    )
                    return
                 
                point.positions = [
                    self.current_joint1_pos,
                    self.current_joint2_pos,
                    self.current_joint3_pos,
                    self.current_joint4_pos,
                    self.current_joint5_pos,
                    self.current_joint6_pos
                ]
                point.time_from_start.sec = 2
                traj.points.append(point)
                self.arm_pub.publish(traj)
                self.get_logger().info(f"✅ Joint movement completed: {point.positions}")
                return

            self.get_logger().warn(f"⚠️ Unknown action: {action}")

        except Exception as e:
            self.get_logger().error(f"❌ process_command error: {e}")

    def _start_keep(self, direction: str):
        d = (direction or "").lower()
        if d not in ("left", "right"):
            self.get_logger().warn(f"⚠️ Unsupported direction for continuous rotate: {direction}. Please use 'left' or 'right'.")
            return

        self._stop_keep()
        self._keep_dir = d
        self._keep_timer = self.create_timer(KEEP_DT, self._on_keep_tick)
        self.get_logger().info(f"🔄 Continuous rotate '{d}' at {KEEP_ROTATE_SPEED_DEG_S:.1f} deg/s (say 'stop' to halt).")

    def _stop_keep(self):
        if self._keep_timer is not None:
            self._keep_timer.cancel()
            self._keep_timer = None
        self._keep_dir = None

    def _on_keep_tick(self):
        if not self._keep_dir:
            return

        traj = JointTrajectory()
        traj.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        pt = JointTrajectoryPoint()

        w = self._keep_w_rad_s
        d = self._keep_dir

        new_positions = [
            self.current_joint1_pos,
            self.current_joint2_pos,
            self.current_joint3_pos,
            self.current_joint4_pos,
            self.current_joint5_pos,
            self.current_joint6_pos
        ]

        if d == "left":
            new_positions[0] += w * KEEP_DT
        elif d == "right":
            new_positions[0] -= w * KEEP_DT

        self.current_joint1_pos = new_positions[0]

        pt.positions = new_positions
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = int(KEEP_DT * 1e9)
        traj.points.append(pt)
        self.arm_pub.publish(traj)

    def get_delta(self, value, unit, radius):
        if unit and unit.lower() in ("degree", "deg"):
            return math.radians(float(value))
        elif unit and unit.lower() == "cm":
            return (float(value) / 100.0) / radius if radius != 0 else 0.0
        elif unit and unit.lower() in ("mm",):
            return ((float(value) / 1000.0) / radius) if radius != 0 else 0.0
        elif unit and unit.lower() in ("m",):
            return (float(value) / radius) if radius != 0 else 0.0
        elif unit and unit.lower() in ("inch", "in"):
            return ((float(value) * 0.0254) / radius) if radius != 0 else 0.0
        return 0.0

    def inverse_kinematics_z(self, z_target):
        delta = z_target - L2
        if abs(delta / L3) > 1.0:
            return False, 0.0
        return True, math.asin(delta / L3)

    def reset_pose(self):
        self.current_joint1_pos = 0.0
        self.current_joint2_pos = 0.0
        self.current_joint3_pos = 0.0
        self.current_joint4_pos = 0.0
        self.current_joint5_pos = 0.0
        self.current_joint6_pos = 0.0
        self.current_z = L2

        traj = JointTrajectory()
        traj.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        pt = JointTrajectoryPoint()
        pt.positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        pt.time_from_start.sec = 2
        traj.points.append(pt)
        self.arm_pub.publish(traj)

        goal = GripperCommand.Goal()
        goal.command = GripperCommandMsg()
        goal.command.position = 0.00
        goal.command.max_effort = 1.0
        self.gripper_client.wait_for_server()
        self.gripper_client.send_goal_async(goal)

        self.get_logger().info("✅ Initialization complete")

    def send_ik_request(self, x, y, z, orientation=None):
        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)

        if orientation is not None:
            pose.pose.orientation = orientation
        else:
            pose.pose.orientation.w = 1.0

        # 🎯 목표 위치 로그
        self.get_logger().info(
            f"🎯 Target → x:{x:.3f}, y:{y:.3f}, z:{z:.3f}, "
            f"ori=({pose.pose.orientation.x:.2f}, {pose.pose.orientation.y:.2f}, "
            f"{pose.pose.orientation.z:.2f}, {pose.pose.orientation.w:.2f})"
        )

        req = GetPositionIK.Request()
        req.ik_request.group_name = "arm"
        req.ik_request.ik_link_name = "end_effector_link"
        req.ik_request.pose_stamped = pose
        req.ik_request.timeout.sec = 2

        future = self.ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res = future.result()

        if res and res.error_code.val == 1:
            j = res.solution.joint_state
            joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
            name_to_pos = dict(zip(j.name, j.position))
            joint_positions = [name_to_pos.get(name, 0.0) for name in joint_names]

            traj = JointTrajectory()
            traj.joint_names = joint_names
            pt = JointTrajectoryPoint()
            pt.positions = joint_positions
            pt.time_from_start.sec = 2
            traj.points.append(pt)

            self.arm_pub.publish(traj)
            self.get_logger().info(f"✅ IK-based movement completed: {joint_positions}")
        else:
            code = res.error_code.val if res else -1
            self.get_logger().error(
                f"❌ IK computation failed (code={code}) → target=({x:.3f}, {y:.3f}, {z:.3f})"
            )

def _spin_worker(node, stop_evt):
    while rclpy.ok() and not stop_evt.is_set():
        rclpy.spin_once(node, timeout_sec=0.05)

def main():
    rclpy.init()
    node = NaturalCommandNode()

    stop_evt = threading.Event()
    spin_thread = threading.Thread(target=_spin_worker, args=(node, stop_evt), daemon=True)
    spin_thread.start()

    print("💬 Enter a command ")
    try:
        while rclpy.ok():
            text = input(">>> ").strip()
            if text.lower() == "exit":
                break

            cmd = node.parse_command_with_llm(text)
            if not cmd:
                print("⚠️ Could not parse command (LLM off or invalid).")
                continue

            print("📦 Parsed command:", json.dumps(cmd, ensure_ascii=False))
            node.process_command(cmd)
    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()
        spin_thread.join(timeout=1.0)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
