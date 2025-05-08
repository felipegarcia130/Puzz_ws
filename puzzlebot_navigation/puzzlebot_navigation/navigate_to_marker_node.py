import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from simple_pid import PID

class FollowLineNode(Node):
    def __init__(self):
        super().__init__('follow_line_node')

        self.publisher = self.create_publisher(Twist, '/cmd_vel_safe', 10)
        self.subscription = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.bridge = CvBridge()

        # PID ajustado para curvas cerradas
        max_yaw = math.radians(30)
        self.max_thr = 0.07  # Cámara lenta → velocidad reducida
        self.yaw_pid = PID(Kp=0.35, Ki=0, Kd=0.1, setpoint=0.0,
                           output_limits=(-max_yaw, max_yaw))

        self.get_logger().info('Seguidor de línea con lógica adaptativa (centrado vs. curva).')

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        frame_height, frame_width = frame.shape[:2]

        # Procesamiento de imagen
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY_INV)
        mask[:int(frame_height * 0.70), :] = 0
        mask[int(frame_height * 0.95):, :] = 0
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=3)

        # Contornos válidos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 100]

        twist = Twist()

        if contours:
            if len(contours) >= 3:
                # Elegir contorno más centrado
                def center_key(c):
                    x, y, w, h = cv2.boundingRect(c)
                    center_x = x + w / 2
                    center_error = abs(center_x - (frame_width / 2))
                    return center_error
                best_contour = sorted(contours, key=center_key)[0]

            else:
                # Elegir contorno más bajo (más cercano)
                def bottom_key(c):
                    x, y, w, h = cv2.boundingRect(c)
                    return -(y + h)
                best_contour = sorted(contours, key=bottom_key)[0]

            # Cálculo del centro del contorno
            M = cv2.moments(best_contour)
            cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else frame_width // 2
            normalized_error = (cx - (frame_width / 2)) / (frame_width / 2)

            # PID yaw
            yaw = self.yaw_pid(normalized_error)

            # Alineación y throttle
            alignment = 1 - abs(normalized_error)
            align_thres = 0.4
            throttle = self.max_thr * ((alignment - align_thres) / (1 - align_thres)) if alignment >= align_thres else 0

            twist.linear.x = float(throttle)
            twist.angular.z = float(yaw)

        self.publisher.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = FollowLineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
