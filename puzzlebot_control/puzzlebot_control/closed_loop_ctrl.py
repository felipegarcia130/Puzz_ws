import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import PoseStamped, Twist
from simple_pid import PID
import math

class ClosedLoopController(Node):
    def __init__(self):
        super().__init__('closed_loop_ctrl')

        # Publisher to /cmd_vel
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # 1) Declare parameters
        self.declare_parameter('Kp_heading', 10.0)
        self.declare_parameter('Ki_heading', 15.0)
        self.declare_parameter('Kd_heading', 0.5)

        self.declare_parameter('Kp_distance', 2.0)
        self.declare_parameter('Ki_distance', 0.0)
        self.declare_parameter('Kd_distance', 0.0)

        self.declare_parameter('distance_output_limit', 0.2)
        self.declare_parameter('angular_output_limit', 2.0)
        self.declare_parameter('yaw_threshold_deg', 5.0)
        self.declare_parameter('integral_threshold_deg', 2.0)
        self.declare_parameter('proximity_threshold', 0.05)  # meters

        # 2) Read them into internal variables
        self._read_params()

        # 3) Instantiate PID controllers
        self.heading_pid = PID(
            self.Kp_heading, self.Ki_heading, self.Kd_heading,
            setpoint=0.0
        )
        self.distance_pid = PID(
            self.Kp_distance, self.Ki_distance, self.Kd_distance,
            setpoint=0.0
        )
        self.distance_pid.output_limits = (
            -self.distance_output_limit,
             self.distance_output_limit
        )
        self.heading_pid.output_limits = (
            -self.angular_output_limit,
            self.angular_output_limit
        )

        # 4) Watch for parameter changes
        self.add_on_set_parameters_callback(self._on_parameter_update)

        # 5) State
        self.latest_estimated_pose = PoseStamped()
        self.latest_goal_pose = PoseStamped()

        # 6) Subscribers
        self.create_subscription(
            PoseStamped, '/estimated_pose',
            self._estimated_cb, 10
        )
        self.create_subscription(
            PoseStamped, '/goal_pose',
            self._goal_cb, 10
        )

        # 7) Debug printing
        self.create_timer(1, self.print_loop)

        # 8) (Optional) Test toggle—unchanged from before
        # self.toggle_state = False
        # self.create_timer(15.0, self.test_pose_toggle)

        self.get_logger().info('Closed-loop controller initialized.')

        # Control loop
        self.control_timer = self.create_timer(1.0/20.0, self.control_loop)

    def control_loop2(self):
        # 1) Extract current yaw
        ez = self.latest_estimated_pose.pose.orientation.z
        ew = self.latest_estimated_pose.pose.orientation.w
        current_yaw = 2.0 * math.atan2(ez, ew)

        # 2) Extract goal yaw
        gz = self.latest_goal_pose.pose.orientation.z
        gw = self.latest_goal_pose.pose.orientation.w
        goal_yaw = 2.0 * math.atan2(gz, gw)

        # 3) Compute shortest‐path angular error
        error = goal_yaw - current_yaw
        error = (error + math.pi) % (2.0 * math.pi) - math.pi

        # convert threshold to radians
        thr = math.radians(self.integral_threshold_deg)
        abs_err = abs(error)

        # compute a 0→1 blend factor: 0 when |error|==thr, 1 when |error|==0
        blend = max(0.0, 1.0 - (abs_err / thr)) if abs_err < thr else 0.0

        # scale Ki by that blend
        Ki_eff = self.Ki_heading * blend

        # re‑tune your PID
        self.heading_pid.tunings = (
            self.Kp_heading,
            Ki_eff,
            self.Kd_heading
        )

        # run PID
        angular_speed = -self.heading_pid(error)

        # publish
        cmd = Twist()
        cmd.linear.x  = 0.0
        cmd.angular.z = angular_speed
        self.cmd_vel_pub.publish(cmd)

    def control_loop(self):
        # 1) Extract current pose
        ex = self.latest_estimated_pose.pose.position.x
        ey = self.latest_estimated_pose.pose.position.y
        ez = self.latest_estimated_pose.pose.orientation.z
        ew = self.latest_estimated_pose.pose.orientation.w
        current_yaw = 2.0 * math.atan2(ez, ew)

        # 2) Extract goal pose
        gx = self.latest_goal_pose.pose.position.x
        gy = self.latest_goal_pose.pose.position.y

        # 3) Compute displacement & distance
        dx = gx - ex
        dy = gy - ey
        distance = math.hypot(dx, dy)

        # 4) Compute angle to goal & heading error
        angle_to_goal = math.atan2(dy, dx)
        error = angle_to_goal - current_yaw
        error = (error + math.pi) % (2.0 * math.pi) - math.pi

        # 4) Blend Ki based on error magnitude
        thr = math.radians(self.integral_threshold_deg)
        abs_err = abs(error)
        blend = (1.0 - abs_err / thr) if abs_err < thr else 0.0
        Ki_eff = self.Ki_heading * blend

        # 5) Re‐tune heading PID on‐the‐fly
        self.heading_pid.tunings = (
            self.Kp_heading,
            Ki_eff,
            self.Kd_heading
        )

        # 6) Point towards goal if far enough away
        angular_speed = -self.heading_pid(error) if distance >= self.proximity_threshold else 0.0

        # 7) Compute distance & scaling factor
        dx = gx - ex
        dy = gy - ey
        distance = math.hypot(dx, dy)
        factor = max(0.0, 1.0 - abs(error) / math.radians(self.yaw_threshold_deg)) if distance >= self.proximity_threshold else 1.0

        # 8) Move towards goal
        linear_speed = factor * (-self.distance_pid(distance))

        # 9) Publish both
        cmd = Twist()
        cmd.linear.x  = linear_speed
        cmd.angular.z = angular_speed
        self.cmd_vel_pub.publish(cmd)

    def test_pose_toggle2(self):
        self.toggle_state = not self.toggle_state

        # Alternate between 0 and 90 degrees
        yaw_deg = 180.0 if self.toggle_state else 0.0
        yaw_rad = math.radians(yaw_deg)

        new_goal = PoseStamped()
        new_goal.pose.position.x = 0.0
        new_goal.pose.position.y = 0.0
        # Encode yaw into z,w quaternion for a planar rotation
        new_goal.pose.orientation.z = math.sin(yaw_rad / 2.0)
        new_goal.pose.orientation.w = math.cos(yaw_rad / 2.0)

        self.latest_goal_pose = new_goal

    def test_pose_toggle(self):
        self.toggle_state = not self.toggle_state

        # Alternate X
        x_value = 1.0 if self.toggle_state else 0.0

        new_goal = PoseStamped()
        new_goal.pose.position.x = x_value
        new_goal.pose.position.y = 0.0

        # Keep orientation at zero yaw
        new_goal.pose.orientation.z = 0.0
        new_goal.pose.orientation.w = 1.0

        self.latest_goal_pose = new_goal

    def _read_params(self):
        """Read all parameters into self.*."""
        p = self.get_parameter
        self.Kp_heading  = p('Kp_heading').value
        self.Ki_heading  = p('Ki_heading').value
        self.Kd_heading  = p('Kd_heading').value
        self.Kp_distance = p('Kp_distance').value
        self.Ki_distance = p('Ki_distance').value
        self.Kd_distance = p('Kd_distance').value
        self.distance_output_limit = p('distance_output_limit').value
        self.angular_output_limit = p('angular_output_limit').value
        self.yaw_threshold_deg     = p('yaw_threshold_deg').value
        self.integral_threshold_deg = p('integral_threshold_deg').value
        self.proximity_threshold = p('proximity_threshold').value

    def _on_parameter_update(self, params):
        """Callback whenever a parameter is set—reconfigure PIDs."""
        updated = []
        for param in params:
            name = param.name
            if name in (
                'Kp_heading','Ki_heading','Kd_heading',
                'Kp_distance','Ki_distance','Kd_distance',
                'distance_output_limit','yaw_threshold_deg',
                'angular_output_limit', 'integral_threshold_deg',
                'proximity_threshold'
            ):
                setattr(self, name, param.value)
                updated.append(name)

        # Re-tune controllers if any relevant param changed
        if any(n.startswith(('Kp_','Ki_','Kd_')) for n in updated):
            self.heading_pid.tunings = (
                self.Kp_heading, self.Ki_heading, self.Kd_heading
            )
            self.distance_pid.tunings = (
                self.Kp_distance, self.Ki_distance, self.Kd_distance
            )
        if 'distance_output_limit' in updated:
            self.distance_pid.output_limits = (
                -self.distance_output_limit,
                 self.distance_output_limit
            )
        if 'angular_output_limit' in updated:
            # self.angular_output_limit = self.get_parameter('angular_output_limit').value
            self.heading_pid.output_limits = (
                -self.angular_output_limit,
                self.angular_output_limit
            )

        return SetParametersResult(successful=True)

    def _estimated_cb(self, msg: PoseStamped):
        self.latest_estimated_pose = msg

    def _goal_cb(self, msg: PoseStamped):
        self.latest_goal_pose = msg

    def print_loop(self):
        def pose_str(label, ps):
            x = ps.pose.position.x
            y = ps.pose.position.y
            qz = ps.pose.orientation.z
            qw = ps.pose.orientation.w
            theta = 2 * math.atan2(qz, qw) * 180 / math.pi
            return f"{label}: x={x:.2f} | y={y:.2f} | θ={theta:.2f}°"

        # Print poses
        print(pose_str("Est", self.latest_estimated_pose))
        print(pose_str("Goal", self.latest_goal_pose))

        # Print all PID & control parameters
        print(
            f"[Params] "
            f"Kp_h={self.Kp_heading:.3f} Ki_h={self.Ki_heading:.3f} Kd_h={self.Kd_heading:.3f} | "
            f"Limit={self.angular_output_limit:.3f} rad/s | I‑thr={self.integral_threshold_deg:.1f}° | "
            f"Kp_d={self.Kp_distance:.3f} Ki_d={self.Ki_distance:.3f} Kd_d={self.Kd_distance:.3f} | "
            f"Limit={self.distance_output_limit:.3f} m/s | "
            f"YawTh={self.yaw_threshold_deg:.1f}°"
        )

def main(args=None):
    rclpy.init(args=args)
    node = ClosedLoopController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
