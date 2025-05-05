"""import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

class Semaforo(Node):
    def __init__(self):
        super().__init__('semaforo_node')
        self.bridge = CvBridge()

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.debug_pub = self.create_publisher(Image, '/debug_image', 10)
        self.mask_pub = self.create_publisher(Image, '/mask_debug', 10)
        self.complete_pub = self.create_publisher(Bool, 'completed_point', 10)

        self.subscription_img = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.subscription_pose = self.create_subscription(PoseStamped, '/estimated_pose', self.pose_callback, 10)
        self.subscription_goal = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)

        self.current_pose = None
        self.current_goal = None
        self.goal_tolerance = 0.15

        self.mission_started = False
        self.frames_since_green = 0
        self.max_linear_vel = 0.2
        self.max_angular_vel = 1.0

        # Parámetros del controlador PI
        self.kp = 0.005
        self.ki = 0.0005
        self.integral_error = 0.0
        self.prev_error = 0.0

    def pose_callback(self, msg):
        self.current_pose = msg
        self.check_goal_reached()

    def goal_callback(self, msg):
        self.current_goal = msg
        self.get_logger().info('🎯 Nuevo objetivo recibido.')

    def check_goal_reached(self):
        if self.current_pose is None or self.current_goal is None:
            return
        dx = self.current_goal.pose.position.x - self.current_pose.pose.position.x
        dy = self.current_goal.pose.position.y - self.current_pose.pose.position.y
        dist = math.hypot(dx, dy)
        if dist < self.goal_tolerance:
            self.get_logger().info('✅ Objetivo alcanzado. Confirmando...')
            self.complete_pub.publish(Bool(data=True))
            self.current_goal = None
            self.mission_started = False

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Rango HSV
        lower_green = np.array([45, 80, 40])
        upper_green = np.array([85, 255, 255])
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        # Máscaras
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_red = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red1, upper_red1),
            cv2.inRange(hsv, lower_red2, upper_red2)
        )
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Publicar máscara para debug
        self.mask_pub.publish(self.bridge.cv2_to_imgmsg(mask_green, encoding='mono8'))

        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        twist = Twist()
        slow_mode = False

        if contours_yellow:
            if cv2.contourArea(max(contours_yellow, key=cv2.contourArea)) > 500:
                slow_mode = True
                self.get_logger().info('🟡 Amarillo detectado: MODO LENTO')

        if contours_red:
            if cv2.contourArea(max(contours_red, key=cv2.contourArea)) > 500:
                self.get_logger().info('🟥 Rojo detectado: DETENIÉNDOSE')
                self.mission_started = False
                self.frames_since_green = 0
                self.integral_error = 0.0  # Reset PI
                self.cmd_pub.publish(Twist())
                return

        if contours_green:
            largest = max(contours_green, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area > 800:
                M = cv2.moments(largest)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    error_x = cx - frame.shape[1] // 2

                    # Controlador PI
                    self.integral_error += error_x
                    angular_z = -(self.kp * error_x + self.ki * self.integral_error)
                    angular_z = max(min(angular_z, self.max_angular_vel), -self.max_angular_vel)

                    twist.linear.x = 0.02 if slow_mode else 0.05
                    twist.linear.x = min(twist.linear.x, self.max_linear_vel)
                    twist.angular.z = angular_z

                    self.mission_started = True
                    self.frames_since_green = 0

                    cv2.drawContours(frame, [largest], -1, (0, 255, 0), 2)
                    cy = int(M['m01'] / M['m00'])
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        elif self.mission_started:
            twist.linear.x = 0.02 if slow_mode else 0.05
            twist.angular.z = 0.0
            self.get_logger().info('⚠️ Verde perdido: AVANZANDO RECTO...')
        else:
            self.integral_error = 0.0  # Evitar acumulación fuera de misión

        self.cmd_pub.publish(twist)
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(frame, encoding='bgr8'))

def main(args=None):
    rclpy.init(args=args)
    node = Semaforo()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class Semaforo(Node):
    def __init__(self):
        super().__init__('semaforo_node')
        self.bridge = CvBridge()
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.debug_pub = self.create_publisher(Image, '/debug_image', 10)
        self.mask_pub = self.create_publisher(Image, '/mask_debug', 10)
        self.subscription = self.create_subscription(Image, '/image_raw', self.image_callback, 10)

        self.mission_started = False
        self.frames_since_green = 0
        self.max_linear_vel = 0.2
        self.max_angular_vel = 1.0

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Rango de colores HSV
        lower_green = np.array([45, 80, 40])
        upper_green = np.array([85, 255, 255])

        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        # Máscaras
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
        slow_mode = False

        # 🟡 Amarillo detectado → ir lento
        if contours_yellow:
            largest_yellow = max(contours_yellow, key=cv2.contourArea)
            if cv2.contourArea(largest_yellow) > 500:
                slow_mode = True
                self.get_logger().info('🟡 Amarillo detectado: MODO LENTO ACTIVADO')

        # 🟥 Rojo detectado → detener misión
        if contours_red:
            largest_red = max(contours_red, key=cv2.contourArea)
            if cv2.contourArea(largest_red) > 500:
                self.get_logger().info('🟥 Rojo detectado: DETENIÉNDOSE')
                self.mission_started = False
                self.frames_since_green = 0
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_pub.publish(twist)
                return

        # 🟢 Verde detectado → iniciar misión
        if contours_green:
            largest = max(contours_green, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            # 🟢 Verde detectado → iniciar misión y moverse en círculo
            if area > 800:
                self.get_logger().info(f'🟢 Verde detectado | Área: {area:.2f}')
                
                twist.linear.x = 0.05 if not slow_mode else 0.03
                twist.angular.z = 0.3 if not slow_mode else 0.15  # Giro constante
                
                self.mission_started = True
                self.frames_since_green = 0

                # Dibujar el contorno
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.drawContours(frame, [largest], -1, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        elif self.mission_started:
            # Sigue girando en círculo
            twist.linear.x = 0.05 if not slow_mode else 0.03
            twist.angular.z = 0.3 if not slow_mode else 0.15
            self.get_logger().info('🟢 Verde perdido: SIGUE CÍRCULO...')

        else:
            # Esperando a que aparezca verde
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.cmd_pub.publish(twist)
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(frame, encoding='bgr8'))

def main(args=None):
    rclpy.init(args=args)
    node = Semaforo()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

