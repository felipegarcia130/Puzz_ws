import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import subprocess
import os

class TeleopJoystick(Node):
    def __init__(self):
        super().__init__('teleop_joystick_node')
        self.subscription = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel_safe', 10)
        self.camera_pub = self.create_publisher(Bool, '/launch_camera', 10)


        self.max_linear = 0.8
        self.max_angular = 2.0
        self.camera_process = None

    def joy_callback(self, msg):
        twist = Twist()
        raw_linear = msg.axes[1]
        raw_angular = msg.axes[3]

        twist.linear.x = max(min(raw_linear * self.max_linear, self.max_linear), -self.max_linear)
        twist.angular.z = max(min(raw_angular * self.max_angular, self.max_angular), -self.max_angular)
        self.publisher.publish(twist)

        # Botón X (índice 0)
        if msg.buttons[0] == 1:
            self.camera_pub.publish(Bool(data=True))  # Activar cámara

        if msg.buttons[1] == 1:
            self.camera_pub.publish(Bool(data=False))  # Apagar cámara


def main(args=None):
    rclpy.init(args=args)
    node = TeleopJoystick()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
