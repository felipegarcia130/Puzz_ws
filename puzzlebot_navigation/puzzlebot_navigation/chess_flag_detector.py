#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class ChessboardByLines(Node):
    def __init__(self):
        super().__init__('chess_flag_hough_detector')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_safe', 10)
        self.debug_pub = self.create_publisher(Image, '/debug_chessboard', 10)
        self.flag_detected = False

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detectar líneas rectas
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                                minLineLength=30, maxLineGap=5)

        vertical = 0
        horizontal = 0

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                if abs(angle) < 15:  # Horizontal
                    horizontal += 1
                elif abs(angle) > 75:  # Vertical
                    vertical += 1
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if horizontal >= 3 and vertical >= 3 and not self.flag_detected:
            self.get_logger().info("🚩 ¡Bandera tipo ajedrez detectada por líneas!")
            self.flag_detected = True
            self.stop_robot()

        # Publicar imagen debug
        debug_img = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.debug_pub.publish(debug_img)

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = ChessboardByLines()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()