

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class Semaforo(Node):
    def __init__(self):
        super().__init__('semaforo_node')
        self.bridge = CvBridge()

        self.mission_pub = self.create_publisher(Bool, '/mission_control', 10)
        self.slow_down_pub = self.create_publisher(Bool, '/slow_down', 10)
        self.mask_pub = self.create_publisher(Image, '/mask_debug', 10)
        self.debug_pub = self.create_publisher(Image, '/debug_image', 10)

        self.subscription = self.create_subscription(Image, '/image_raw', self.image_callback, 10)

        self.mission_started = False
        self.waiting_for_green = False  # 🔴 Nueva bandera de espera

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Rangos HSV
        lower_green = np.array([65, 40, 100])
        upper_green = np.array([90, 255, 255])

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

        kernel = np.ones((5, 5), np.uint8)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)

        self.mask_pub.publish(self.bridge.cv2_to_imgmsg(mask_green, encoding='mono8'))

        # Contornos
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 🔴 DETECCIÓN DE ROJO
        if contours_red:
            largest_red = max(contours_red, key=cv2.contourArea)
            if cv2.contourArea(largest_red) > 300:
                self.get_logger().info('🟥 Rojo detectado → DETENIÉNDOSE y esperando verde')
                self.mission_pub.publish(Bool(data=False))
                self.slow_down_pub.publish(Bool(data=False))
                self.mission_started = False
                self.waiting_for_green = True
                return  # No procesar amarillo ni verde

        # Si está esperando verde, solo lo detecta y reanuda misión
        if self.waiting_for_green:
            if contours_green:
                largest_green = max(contours_green, key=cv2.contourArea)
                if cv2.contourArea(largest_green) > 300:
                    self.get_logger().info('🟢 Verde detectado → REANUDANDO misión')
                    self.mission_pub.publish(Bool(data=True))
                    self.slow_down_pub.publish(Bool(data=False))
                    self.mission_started = True
                    self.waiting_for_green = False
            return  # Mientras esté esperando verde, ignora el resto

        # 🟨 Amarillo: baja velocidad si ya está en movimiento
        if contours_yellow and self.mission_started:
            largest_yellow = max(contours_yellow, key=cv2.contourArea)
            if cv2.contourArea(largest_yellow) > 300:
                self.get_logger().info('🟨 Amarillo detectado → REDUCIENDO VELOCIDAD')
                self.slow_down_pub.publish(Bool(data=True))
                return

        # 🟢 Verde: misión activa y velocidad normal
        if contours_green:
            largest_green = max(contours_green, key=cv2.contourArea)
            if cv2.contourArea(largest_green) > 300:
                if not self.mission_started:
                    self.get_logger().info('🟢 Verde detectado → INICIANDO MISIÓN')
                    self.mission_pub.publish(Bool(data=True))
                    self.mission_started = True
                self.slow_down_pub.publish(Bool(data=False))

        # Si no hay colores válidos detectados
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(frame, encoding='bgr8'))

def main(args=None):
    rclpy.init(args=args)
    node = Semaforo()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


"""import rclpy
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
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_safe', 10)
        self.debug_pub = self.create_publisher(Image, '/debug_image', 10)
        self.mask_pub = self.create_publisher(Image, '/mask_debug', 10)
        self.subscription = self.create_subscription(Image, '/image_raw', self.image_callback, 10)

        self.mission_started = False
        self.frames_since_green = 0
        self.max_linear_vel = 0.5
        self.max_angular_vel = 1.0

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Rango de colores HSV
        lower_green = np.array([35, 40, 30])   # Verde más oscuro, menos saturado
        upper_green = np.array([90, 255, 255]) # Verde muy brillante, incluso tirando a blanco


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
    main()"""
