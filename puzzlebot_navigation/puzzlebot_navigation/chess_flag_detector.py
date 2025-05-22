import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class ChessFlagDetector(Node):
    def __init__(self):
        super().__init__('chess_flag_detector')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_safe', 10)
        self.debug_pub = self.create_publisher(Image, '/debug_chessboard', 10)
        self.flag_detected = False

        # Tamaño del patrón: número de esquinas internas (horizontal x vertical)
        self.pattern_size = (6, 5)  # Ajusta si tu tablero es diferente

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Buscar patrón de ajedrez
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)

        if ret and not self.flag_detected:
            self.get_logger().info("🚩 ¡Bandera de meta detectada!")
            self.flag_detected = True
            self.stop_robot()

            # Dibujar el patrón
            cv2.drawChessboardCorners(frame, self.pattern_size, corners, ret)
        else:
            if not self.flag_detected:
                self.get_logger().info("Buscando bandera...")

        # Publicar imagen de depuración
        debug_img = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.debug_pub.publish(debug_img)

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = ChessFlagDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
