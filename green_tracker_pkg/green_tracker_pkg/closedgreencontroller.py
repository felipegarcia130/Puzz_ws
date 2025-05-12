import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Bool
import math

class ClosedLoopController(Node):
    def __init__(self):
        super().__init__('closed_loop_controller')

        # Parámetros PID
        self.declare_parameter('Kp_lin', 1.0)
        self.declare_parameter('Ki_lin', 0.0)
        self.declare_parameter('Kd_lin', 0.1)
        self.declare_parameter('Kp_ang', 2.0)
        self.declare_parameter('Ki_ang', 0.0)
        self.declare_parameter('Kd_ang', 0.1)
        self.declare_parameter('MAX_V', 0.2)
        self.declare_parameter('MAX_W', 0.2)
        self.declare_parameter('TOL_POS', 0.05)
        self.declare_parameter('TOL_ANG', 0.1)

        self.Kp_lin = self.get_parameter('Kp_lin').value
        self.Ki_lin = self.get_parameter('Ki_lin').value
        self.Kd_lin = self.get_parameter('Kd_lin').value
        self.Kp_ang = self.get_parameter('Kp_ang').value
        self.Ki_ang = self.get_parameter('Ki_ang').value
        self.Kd_ang = self.get_parameter('Kd_ang').value
        self.MAX_V = self.get_parameter('MAX_V').value
        self.MAX_W = self.get_parameter('MAX_W').value
        self.TOL_POS = self.get_parameter('TOL_POS').value
        self.TOL_ANG = self.get_parameter('TOL_ANG').value

        self.goal_pose = None
        self.current_pose = None
        self.mission_active = False  # 🔑 NUEVO: semáforo
        self.last_time = self.get_clock().now()
        self.e_sum_lin = 0.0
        self.e_last_lin = 0.0
        self.e_sum_ang = 0.0
        self.e_last_ang = 0.0
        self.reached_sent = False

        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.create_subscription(PoseStamped, '/estimated_pose', self.pose_callback, 10)
        self.create_subscription(Bool, '/mission_control', self.mission_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_safe', 10)
        self.reached_pub = self.create_publisher(Bool, 'completed_point', 10)

        self.control_timer = self.create_timer(0.02, self.control_loop)

        self.get_logger().info('🚀 Closed-loop controller listo.')

    def mission_callback(self, msg: Bool):
        self.mission_active = msg.data
        estado = "🟢 ACTIVADO" if msg.data else "🔴 DETENIDO"
        self.get_logger().info(f'[Semáforo] Misión: {estado}')

    def goal_callback(self, msg):
        self.goal_pose = msg
        self.e_sum_lin = 0.0
        self.e_last_lin = 0.0
        self.e_sum_ang = 0.0
        self.e_last_ang = 0.0
        self.reached_sent = False

    def pose_callback(self, msg):
        self.current_pose = msg

    def control_loop(self):
        if not self.mission_active:
            self.cmd_pub.publish(Twist())  # Detener
            return

        if self.goal_pose is None or self.current_pose is None:
            return

        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now

        xr = self.current_pose.pose.position.x
        yr = self.current_pose.pose.position.y
        _, _, thetar = self.euler_from_quaternion(self.current_pose.pose.orientation)

        xg = self.goal_pose.pose.position.x
        yg = self.goal_pose.pose.position.y
        _, _, thetag = self.euler_from_quaternion(self.goal_pose.pose.orientation)

        dx = xg - xr
        dy = yg - yr
        rho = math.hypot(dx, dy)
        path_theta = math.atan2(dy, dx)
        e_theta = self.normalize_angle(path_theta - thetar)
        e_final = self.normalize_angle(thetag - thetar)

        twist = Twist()

        if rho > self.TOL_POS:
            e_lin = rho
            self.e_sum_lin += e_lin * dt
            d_lin = (e_lin - self.e_last_lin) / dt if dt > 0 else 0.0
            v = self.Kp_lin * e_lin + self.Ki_lin * self.e_sum_lin + self.Kd_lin * d_lin
            self.e_last_lin = e_lin

            e_ang = e_theta
            self.e_sum_ang += e_ang * dt
            d_ang = (e_ang - self.e_last_ang) / dt if dt > 0 else 0.0
            w = self.Kp_ang * e_ang + self.Ki_ang * self.e_sum_ang + self.Kd_ang * d_ang
            self.e_last_ang = e_ang

            twist.linear.x = max(-self.MAX_V, min(self.MAX_V, v))
            twist.angular.z = max(-self.MAX_W, min(self.MAX_W, w))
            self.reached_sent = False
        else:
            if abs(e_final) > self.TOL_ANG:
                w = self.Kp_ang * e_final
                twist.linear.x = 0.0
                twist.angular.z = max(-self.MAX_W, min(self.MAX_W, w))
                self.reached_sent = False
            else:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                if not self.reached_sent:
                    self.get_logger().info('✅ Objetivo alcanzado. Enviando confirmación...')
                    self.reached_pub.publish(Bool(data=True))
                    self.reached_sent = True
                    self.goal_pose = None

        self.cmd_pub.publish(twist)

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def euler_from_quaternion(self, q):
        x, y, z, w = q.x, q.y, q.z, q.w
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return 0.0, 0.0, yaw

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
