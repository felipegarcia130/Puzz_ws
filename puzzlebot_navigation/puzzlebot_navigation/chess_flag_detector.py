#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np

class ChessboardFlagDistanceNode(Node):
    def __init__(self):
        super().__init__('chessboard_flag_distance_node')

        # Cámara intrínseca y distorsión
        self.K = np.array([
            [394.32766428, 0.0, 343.71433623],
            [0.0, 524.94987967, 274.24900983],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        self.D = np.array([-0.02983132, -0.02312677, 0.03447185, -0.02105932], dtype=np.float64)
        self.f_y = self.K[1, 1]

        # Parámetros del checkerboard
        self.pattern_size = (4, 3)
        self.square_size = 0.025  # m
        self.threshold = 0.40     # m

        # Utilidades
        self.bridge = CvBridge()

        # Subscripción a la cámara
        self.image_sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)

        # Publicadores
        self.flag_pub = self.create_publisher(Bool, '/flag_close', 10)
        self.debug_pub = self.create_publisher(Image, '/debug_chessboard', 10)

        self.get_logger().info("Nodo de detección de bandera tipo ajedrez iniciado.")

    def undistort_frame(self, frame):
        return cv2.undistort(frame, self.K, self.D)

    def get_checkerboard_corners(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
        return corners if ret else None

    def estimate_distance_from_height(self, corners):
        ys = corners[:, :, 1].flatten()
        h_pix = ys.max() - ys.min()
        H_real = self.square_size * (self.pattern_size[1] - 1)
        Z = (self.f_y * H_real) / h_pix
        return Z, h_pix

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        undist = self.undistort_frame(frame)
        corners = self.get_checkerboard_corners(undist)

        flag_close = False
        if corners is not None:
            Z, _ = self.estimate_distance_from_height(corners)
            flag_close = Z <= self.threshold
            cv2.drawChessboardCorners(undist, self.pattern_size, corners, True)

        # Publicar imagen debug
        status_text = "FLAG CERCA (<=40cm)" if flag_close else "FLAG LEJOS"
        color = (0, 255, 0) if flag_close else (0, 0, 255)
        cv2.putText(undist, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        debug_msg = self.bridge.cv2_to_imgmsg(undist, encoding='bgr8')
        self.debug_pub.publish(debug_msg)

        # Publicar estado
        self.flag_pub.publish(Bool(data=bool(flag_close)))

def main(args=None):
    rclpy.init(args=args)
    node = ChessboardFlagDistanceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
