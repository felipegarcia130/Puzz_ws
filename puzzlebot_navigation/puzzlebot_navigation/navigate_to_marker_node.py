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

        #self.publisher = self.create_publisher(Twist, '/cmd_vel_safe', 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.debug_pub = self.create_publisher(Image, '/debug_image', 10)
        self.bridge = CvBridge()

        # PID controller
        max_yaw = math.radians(60)
        self.max_thr = 0.2
        self.yaw_pid = PID(Kp=0.6, Ki=0, Kd=0.1, setpoint=0.0, output_limits=(-max_yaw, max_yaw))

        self.get_logger().info('Seguidor de línea con lógica angular avanzada activado.')

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        throttle, yaw = self.follow_line(frame)

        # Publicar comando de movimiento
        twist = Twist()
        twist.linear.x = float(throttle)
        twist.angular.z = float(yaw)
        self.publisher.publish(twist)

        # Publicar imagen procesada para rqt_image_view
        debug_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.debug_pub.publish(debug_msg)

    def follow_line(self, frame):
        if frame is None:
            return 0.0, 0.0

        frame_height, frame_width = frame.shape[:2]

        # Threshold y preprocesamiento
        dark_thres = 100
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, dark_thres, 255, cv2.THRESH_BINARY_INV)
        mask[:int(frame_height * 0.7), :] = 0
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=3)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=5)

        # Contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        MIN_AREA = 1000
        contours = [c for c in contours if cv2.contourArea(c) > MIN_AREA]

        # Mostrar líneas candidatas
        for i, c in enumerate(contours):
            pt1, pt2, angle, cx, cy = self.get_contour_line(c)
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(frame, str(i), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        throttle, yaw = 0.0, 0.0
        if contours:
            def contour_key(c):
                _, _, angle, cx, cy = self.get_contour_line(c)
                max_angle = 80
                angle = max(min(angle, max_angle), -max_angle)
                ref_x = (frame_width / 2) + (angle / max_angle) * (frame_width / 2)
                cv2.line(frame, (int(ref_x), 0), (int(ref_x), frame_height), (0, 0, 255), 2)
                x_err = abs(cx - ref_x)
                return x_err

            line_contour = sorted(contours, key=contour_key)[0]

            # Visualización
            cv2.drawContours(frame, [line_contour], -1, (0, 255, 0), 2)
            for c in contours[1:]:
                cv2.drawContours(frame, [c], -1, (0, 0, 255), 2)

            x, y, w, h = cv2.boundingRect(line_contour)
            center_x = x + w // 2
            normalized_x = (center_x - (frame_width / 2)) / (frame_width / 2)
            cv2.line(frame, (center_x, 0), (center_x, frame_height), (255, 0, 0), 2)

            yaw = self.yaw_pid(normalized_x)
            alignment = 1 - abs(normalized_x)
            align_thres = 0.3
            throttle = self.max_thr * ((alignment - align_thres) / (1 - align_thres)) if alignment >= align_thres else 0
        else:
            cv2.putText(frame, "Searching for line", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return throttle, yaw

    def get_contour_line(self, c, fix_vert=True):
        vx, vy, cx, cy = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        scale = 100
        pt1 = (int(cx - vx * scale), int(cy - vy * scale))
        pt2 = (int(cx + vx * scale), int(cy + vy * scale))
        angle = math.degrees(math.atan2(vy, vx))

        if fix_vert:
            angle = angle - 90 * np.sign(angle)

        return pt1, pt2, angle, cx, cy

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
