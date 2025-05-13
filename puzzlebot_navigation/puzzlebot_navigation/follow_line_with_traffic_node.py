import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from simple_pid import PID

class FollowLineWithTraffic(Node):
    def __init__(self):
        super().__init__('follow_line_with_traffic_node')

        self.bridge = CvBridge()
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_safe', 10)
        self.debug_pub = self.create_publisher(Image, '/debug_image', 10)
        self.mask_pub = self.create_publisher(Image, '/mask_debug', 10)
        self.sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)

        # Estado de misión
        self.mission_started = False
        self.slow_mode = False

        # PID
        self.max_yaw = math.radians(60)
        self.max_thr_normal = 0.2
        self.max_thr_slow = 0.1
        self.yaw_pid = PID(Kp=0.6, Ki=0, Kd=0.1, setpoint=0.0, output_limits=(-self.max_yaw, self.max_yaw))

        self.get_logger().info('Nodo integrado de semáforo y seguidor de línea activo.')

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # === SEMÁFORO ===
        lower_green = np.array([35, 40, 30])
        upper_green = np.array([90, 255, 255])
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        kernel = np.ones((5, 5), np.uint8)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
        self.mask_pub.publish(self.bridge.cv2_to_imgmsg(mask_green, encoding='mono8'))

        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        twist = Twist()
        self.slow_mode = False

        
        if contours_yellow:
            if cv2.contourArea(max(contours_yellow, key=cv2.contourArea)) > 500:
                if self.mission_started:
                    self.slow_mode = True
                    self.get_logger().info('🟡 Amarillo detectado: modo lento.')
                else:
                    self.get_logger().info('🟡 Amarillo detectado pero misión no iniciada. Ignorando.')


        if contours_red:
            if cv2.contourArea(max(contours_red, key=cv2.contourArea)) > 500:
                self.get_logger().info('🟥 Rojo detectado: detención total.')
                self.mission_started = False
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_pub.publish(twist)
                return

        if contours_green:
            if cv2.contourArea(max(contours_green, key=cv2.contourArea)) > 800:
                self.mission_started = True
                self.get_logger().info('🟢 Verde detectado: misión iniciada.')

        if not self.mission_started:
            self.get_logger().info('⏳ Esperando verde...')
            self.cmd_pub.publish(Twist())
            return

        # === SEGUIDOR DE LÍNEA ===
        throttle, yaw = self.follow_line(frame)

        twist.linear.x = float(throttle)
        twist.angular.z = float(yaw)


        self.cmd_pub.publish(twist)
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(frame, encoding="bgr8"))

    def follow_line(self, frame):
        frame_height, frame_width = frame.shape[:2]

        # Línea negra sobre fondo claro
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        mask[:int(frame_height * 0.7), :] = 0
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=3)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=5)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 1000]

        if not contours:
            cv2.putText(frame, "Searching for line", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return 0.0, 0.0

        def get_contour_line(c):
            vx, vy, cx, cy = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            angle = math.degrees(math.atan2(vy, vx))
            angle -= 90 * np.sign(angle)
            return cx, angle

        def contour_key(c):
            cx, angle = get_contour_line(c)
            max_angle = 80
            angle = max(min(angle, max_angle), -max_angle)
            ref_x = (frame_width / 2) + (angle / max_angle) * (frame_width / 2)
            cv2.line(frame, (int(ref_x), 0), (int(ref_x), frame_height), (0, 0, 255), 2)
            return abs(cx - ref_x)

        best_contour = sorted(contours, key=contour_key)[0]
        x, y, w, h = cv2.boundingRect(best_contour)
        center_x = x + w // 2
        cv2.line(frame, (center_x, 0), (center_x, frame_height), (255, 0, 0), 2)
        normalized_x = (center_x - (frame_width / 2)) / (frame_width / 2)

        yaw = self.yaw_pid(normalized_x)
        alignment = 1 - abs(normalized_x)
        align_thres = 0.3
        max_thr = self.max_thr_slow if self.slow_mode else self.max_thr_normal
        throttle = max_thr * ((alignment - align_thres) / (1 - align_thres)) if alignment >= align_thres else 0

        return throttle, yaw

def main(args=None):
    rclpy.init(args=args)
    node = FollowLineWithTraffic()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
