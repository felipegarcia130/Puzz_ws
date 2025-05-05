import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist

class TeleopJoystick(Node):
    def __init__(self):
        super().__init__('teleop_joystick_node')
        self.subscription = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Parámetros máximos de velocidad
        self.max_linear = 0.2    # m/s
        self.max_angular = 0.2   # rad/s

    def joy_callback(self, msg):
        twist = Twist()

        raw_linear = msg.axes[1]
        raw_angular = msg.axes[3]

        # Escalado y recorte de velocidades
        twist.linear.x = max(min(raw_linear * self.max_linear, self.max_linear), -self.max_linear)
        twist.angular.z = max(min(raw_angular * self.max_angular, self.max_angular), -self.max_angular)

        self.publisher.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = TeleopJoystick()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
