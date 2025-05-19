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
        self.debug_pub = self.create_publisher(Image, '/debug_image', 10)
        self.bridge = CvBridge()

        # PID controller
        max_yaw = math.radians(60)
        self.max_thr = 0.2
        self.yaw_pid = PID(Kp=0.6, Ki=0.0, Kd=0.1, setpoint=0.0, output_limits=(-max_yaw, max_yaw))

        self.get_logger().info('Seguidor de línea con lógica adaptativa para curvas y rectas activado.')

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        throttle, yaw = self.follow_line(frame)
        twist = Twist()
        twist.linear.x = float(throttle)
        twist.angular.z = float(yaw)
        self.publisher.publish(twist)
        debug_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.debug_pub.publish(debug_msg)

    def follow_line(self, frame):
        if frame is None:
            return 0.0, 0.0

        frame_height, frame_width = frame.shape[:2]

        # Preprocesamiento de imagen
        dark_thres = 100
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)  # Mejora del contraste general
        _, mask = cv2.threshold(gray_eq, dark_thres, 255, cv2.THRESH_BINARY_INV)

        # Limitar la detección a la parte inferior del frame (por ejemplo, el 30% inferior)
        mask[:int(frame_height * 0.7), :] = 0

        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=3)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=5)

        # Detección de contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        MIN_AREA = 1000
        contours = [c for c in contours if cv2.contourArea(c) > MIN_AREA]

        throttle, yaw = 0.0, 0.0

        if contours:
            # Seleccionar el contorno que tenga mayor "score" (mayor área y más cerca de la parte inferior)
            line_contour = max(contours, key=lambda c: self.contour_score(c, frame_height))

            # Obtener línea ajustada del contorno
            pt1, pt2, angle, cx, cy = self.get_contour_line(line_contour)

            # Dibujo visual de contorno y línea estimada
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            cv2.drawContours(frame, [line_contour], -1, (0, 255, 0), 2)
            for c in contours:
                if not np.array_equal(c, line_contour):
                    cv2.drawContours(frame, [c], -1, (0, 0, 255), 1)

            # Ajuste de referencia basado en si se está en curva o recta
            if self.is_turn(angle):
                ref_x = frame_width * 0.65 if angle > 0 else frame_width * 0.35
            else:
                ref_x = frame_width / 2

            cv2.line(frame, (int(ref_x), 0), (int(ref_x), frame_height), (0, 0, 255), 2)

            # Calcular el centro del contorno usando el rectángulo delimitador
            x, y, w, h = cv2.boundingRect(line_contour)
            center_x = x + w // 2
            normalized_x = (center_x - ref_x) / (frame_width / 2)
            cv2.line(frame, (center_x, 0), (center_x, frame_height), (255, 0, 0), 2)

            yaw = self.yaw_pid(normalized_x)

            alignment = 1 - abs(normalized_x)
            align_thres = 0.3
            curve_factor = max(0.5, 1 - abs(angle) / 90)  # Ajuste para curvas: mínimo 50% del throttle
            throttle = self.max_thr * curve_factor * ((alignment - align_thres) / (1 - align_thres)) if alignment >= align_thres else 0

            # Mensaje de diagnóstico
            text = "CURVA" if self.is_turn(angle) else "RECTA"
            cv2.putText(frame, f"{text} | angle: {angle:.1f}", (10, frame_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        else:
            cv2.putText(frame, "Searching for line", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return throttle, yaw

    def contour_score(self, contour, frame_height):
        """
        Puntuación para el contorno, combinando su área y la posición (más abajo es mejor)
        """
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        # Se favorece contornos más grandes y que estén cerca de la parte inferior del frame
        return area * (y + h)

    def get_contour_line(self, c, fix_vert=True):
        vx, vy, cx, cy = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        scale = 100
        pt1 = (int(cx - vx * scale), int(cy - vy * scale))
        pt2 = (int(cx + vx * scale), int(cy + vy * scale))
        angle = math.degrees(math.atan2(vy, vx))
        if fix_vert:
            angle = angle - 90 * np.sign(angle)
        return pt1, pt2, angle, cx, cy

    def is_turn(self, angle, angle_thresh=25):
        return abs(angle) > angle_thresh

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
