import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped
import math
import time

class CmdVelLimiter(Node):
    def __init__(self):
        super().__init__('cmd_vel_limiter')

        self.max_v = 0.8  # m/s
        self.max_w = 1.0  # rad/s

        self.last_cmd = Twist()
        self.last_time = self.get_clock().now()

        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel_safe',
            self.cmd_callback,
            10
        )

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.robot_vel_pub = self.create_publisher(TwistStamped, '/robot_vel', 10)

        self.get_logger().info('⚙️ Nodo cmd_vel_limiter activo.')

    def cmd_callback(self, msg: Twist):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now

        # Limitar aceleración (lineal y angular)
        max_acc_v = 0.3  # m/s²
        max_acc_w = 2.0  # rad/s²

        v_target = max(-self.max_v, min(self.max_v, msg.linear.x))
        w_target = max(-self.max_w, min(self.max_w, msg.angular.z))

        dv = v_target - self.last_cmd.linear.x
        dw = w_target - self.last_cmd.angular.z

        dv = max(-max_acc_v * dt, min(max_acc_v * dt, dv))
        dw = max(-max_acc_w * dt, min(max_acc_w * dt, dw))

        self.last_cmd.linear.x += dv
        self.last_cmd.angular.z += dw

        # Publicar comando final limitado
        self.cmd_pub.publish(self.last_cmd)

        # Opcional: publicar también como robot_vel
        stamped = TwistStamped()
        stamped.header.stamp = now.to_msg()
        stamped.header.frame_id = 'base_link'
        stamped.twist = self.last_cmd
        self.robot_vel_pub.publish(stamped)

def main(args=None):
    rclpy.init(args=args)
    node = CmdVelLimiter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()