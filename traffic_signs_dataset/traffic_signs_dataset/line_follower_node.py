#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import time

# ==================== CLASE PID ORIGINAL ====================

class SimplePID:
    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0, setpoint=0.0, output_limits=None):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        
        # Variables de estado
        self._integral = 0.0
        self._last_error = 0.0
        self._last_time = None
        
    def __call__(self, input_value):
        current_time = time.time()
        
        # Calcular error
        error = self.setpoint - input_value
        
        # Inicializar tiempo si es la primera llamada
        if self._last_time is None:
            self._last_time = current_time
            self._last_error = error
            return 0.0
        
        # Calcular delta de tiempo
        dt = current_time - self._last_time
        if dt <= 0.0:
            dt = 1e-6  # Evitar división por cero
        
        # Término proporcional
        proportional = self.Kp * error
        
        # Término integral
        self._integral += error * dt
        integral = self.Ki * self._integral
        
        # Término derivativo
        derivative = self.Kd * (error - self._last_error) / dt
        
        # Salida del PID
        output = proportional + integral + derivative
        
        # Aplicar límites de salida
        if self.output_limits is not None:
            min_limit, max_limit = self.output_limits
            if output > max_limit:
                output = max_limit
                # Anti-windup: limitar integral si estamos saturados
                self._integral -= error * dt
            elif output < min_limit:
                output = min_limit
                # Anti-windup: limitar integral si estamos saturados
                self._integral -= error * dt
        
        # Guardar valores para la siguiente iteración
        self._last_error = error
        self._last_time = current_time
        
        return output
    
    def reset(self):
        """Reinicia el controlador PID"""
        self._integral = 0.0
        self._last_error = 0.0
        self._last_time = None

# ==================== MENSAJE PERSONALIZADO (SIMULADO) ====================

class NavigationCommand:
    def __init__(self):
        self.header = None
        self.state = ""
        self.turn_command = ""
        self.authority = 1.0
        self.stopping = False
        self.throttle = 0.0
        self.yaw = 0.0

# ==================== NODO SEGUIDOR DE LÍNEAS ====================

class LineFollowerNode(Node):
    def __init__(self):
        super().__init__('line_follower_node')
        
        # Parámetros
        self.declare_parameter('max_yaw_deg', 60.0)
        self.declare_parameter('max_throttle', 0.2)
        self.declare_parameter('align_threshold', 0.3)
        self.declare_parameter('kp', 0.8)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.15)
        self.declare_parameter('v_fov', 0.4)
        
        max_yaw_deg = self.get_parameter('max_yaw_deg').get_parameter_value().double_value
        self.max_thr = self.get_parameter('max_throttle').get_parameter_value().double_value
        self.align_thres = self.get_parameter('align_threshold').get_parameter_value().double_value
        kp = self.get_parameter('kp').get_parameter_value().double_value
        ki = self.get_parameter('ki').get_parameter_value().double_value
        kd = self.get_parameter('kd').get_parameter_value().double_value
        self.v_fov = self.get_parameter('v_fov').get_parameter_value().double_value
        
        self.max_yaw = math.radians(max_yaw_deg)
        
        # Inicializar PID
        self.yaw_pid = SimplePID(Kp=kp, Ki=ki, Kd=kd, setpoint=0.0, 
                                output_limits=(-self.max_yaw, self.max_yaw))
        
        # Variables de estado del seguidor de líneas original
        self.last_yaw = 0.0
        self.last_thr = 0.05
        self.authority = 1.0  # Factor de autoridad para control externo
        
        # Para corrección de sesgo
        self.last_angle = 0.0
        self.transition_frames = 0
        self.straight_counter = 0
        
        # ROS2 setup
        self.bridge = CvBridge()
        
        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/line_following_cmd', 10)
        self.debug_pub = self.create_publisher(Image, '/line_debug_image', 10)
        self.authority_pub = self.create_publisher(Float32, '/line_authority', 10)
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10)
        self.nav_cmd_sub = self.create_subscription(
            NavigationCommand, '/navigation_command', self.nav_command_callback, 10)
        
        self.get_logger().info('Line Follower Node initialized')
    
    def nav_command_callback(self, msg):
        """Callback para recibir comandos de navegación (authority, etc.)"""
        try:
            self.authority = msg.authority
            
            # Publicar authority actual
            auth_msg = Float32()
            auth_msg.data = self.authority
            self.authority_pub.publish(auth_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing navigation command: {str(e)}')
    
    def enhance_saturation(self, frame, saturation_factor=1.4):
        """Aumentar saturación para colores más vivos"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        s = s.astype(np.float32)
        s = s * saturation_factor
        s = np.clip(s, 0, 255).astype(np.uint8)
        
        enhanced_hsv = cv2.merge([h, s, v])
        enhanced_frame = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
        return enhanced_frame

    def adaptive_thres(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 141, 6)
        return mask

    def get_line_mask(self, frame):
        mask = self.adaptive_thres(frame)
        mask[:int(frame.shape[0] * (1 - self.v_fov)), :] = 0
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=3)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=5)
        return mask

    def get_contour_line_info(self, c):
        vx, vy, cx, cy = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        projections = [((pt[0][0] - cx) * vx + (pt[0][1] - cy) * vy) for pt in c]
        min_proj = min(projections)
        max_proj = max(projections)
        pt1 = (int(cx + vx * min_proj), int(cy + vy * min_proj))
        pt2 = (int(cx + vx * max_proj), int(cy + vy * max_proj))
        angle = math.degrees(math.atan2(vy, vx)) - 90 * np.sign(math.degrees(math.atan2(vy, vx)))
        length = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
        return pt1, pt2, angle, cx, cy, length

    def get_line_candidates(self, frame):
        mask = self.get_line_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 800]
        lines = [self.get_contour_line_info(c) for c in contours]
        return [(c, l) for c, l in zip(contours, lines) if l[5] > 50]

    def get_middle_line(self, frame, drawing_frame=None):
        frame_height, frame_width = frame.shape[:2]

        def line_key(l):
            _, _, angle, cx, _, _ = l[1]
            max_angle = 80
            angle = max(min(angle, max_angle), -max_angle)
            ref_x = (frame_width / 2) + (angle / max_angle) * (frame_width / 2)
            return abs(cx - ref_x)

        lines = self.get_line_candidates(frame)
        if lines:
            lines = sorted(lines, key=line_key)
            best = lines[0]
            if drawing_frame is not None:
                cv2.drawContours(drawing_frame, [best[0]], -1, (0, 255, 0), 2)
                for c, _ in lines[1:]:
                    cv2.drawContours(drawing_frame, [c], -1, (0, 0, 255), 2)
            return best
        return None

    def follow_line(self, frame, drawing_frame=None):
        """Función original de seguimiento de líneas (sin cambios)"""
        line = self.get_middle_line(frame, drawing_frame)
        throttle, yaw = 0.0, 0.0
        frame_height, frame_width = frame.shape[:2]

        if line:
            contour, (pt1, pt2, angle, cx, cy, length) = line
            x, y, w, h = cv2.boundingRect(contour)
            
            # Detectar transición y mantener contador de rectas
            is_straight = abs(angle) < 10
            was_curve = abs(self.last_angle) >= 15
            is_transition = is_straight and was_curve
            
            if is_straight:
                self.straight_counter += 1
            else:
                self.straight_counter = 0
            
            if is_transition:
                self.transition_frames = 12
                self.yaw_pid.reset()
                self.yaw_pid._integral = 0.0
            
            # Para rectas, usar múltiples métodos de cálculo
            if is_straight:
                roi_height = int(frame_height * 0.25)
                roi_y_start = frame_height - roi_height
                
                contour_roi = []
                for point in contour:
                    if point[0][1] >= roi_y_start:
                        contour_roi.append(point)
                
                center_x_roi = x + w // 2
                if len(contour_roi) > 10:
                    contour_roi = np.array(contour_roi)
                    M_roi = cv2.moments(contour_roi)
                    if M_roi["m00"] != 0:
                        center_x_roi = int(M_roi["m10"] / M_roi["m00"])
                
                M_full = cv2.moments(contour)
                center_x_full = x + w // 2
                if M_full["m00"] != 0:
                    center_x_full = int(M_full["m10"] / M_full["m00"])
                
                center_x = int(0.7 * center_x_roi + 0.3 * center_x_full)
            else:
                center_x = x + w // 2
            
            normalized_x = (center_x - (frame_width / 2)) / (frame_width / 2)
            
            # Compensación súper agresiva
            if self.transition_frames > 0:
                if self.transition_frames > 8:
                    strength = 0.25
                elif self.transition_frames > 4:
                    strength = 0.15
                else:
                    strength = 0.08
                    
                bias_correction = -strength
                normalized_x += bias_correction
                self.transition_frames -= 1
                
                if drawing_frame is not None:
                    cv2.putText(drawing_frame, f"STRONG CORRECTION: {strength:.3f}", (10, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            elif is_straight and self.straight_counter > 2:
                continuous_correction = -0.04
                normalized_x += continuous_correction
                
                if drawing_frame is not None:
                    cv2.putText(drawing_frame, "CONTINUOUS CORRECTION", (10, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            if is_straight and abs(normalized_x) < 0.015:
                normalized_x = 0.0
            
            yaw = self.yaw_pid(normalized_x)
            self.last_angle = angle

            alignment = 1 - abs(normalized_x)
            x = ((alignment - self.align_thres) / (1 - self.align_thres))
            throttle = self.max_thr * x

            # Aplicar authority
            throttle *= self.authority
            yaw *= self.authority

            self.last_yaw = yaw
            self.last_thr = throttle

            if drawing_frame is not None:
                cv2.line(drawing_frame, (center_x, 0), (center_x, frame_height), (255, 0, 0), 2)
                cv2.line(drawing_frame, (frame_width//2, 0), (frame_width//2, frame_height), (0, 255, 255), 1)
        else:
            yaw = self.last_yaw
            throttle = 0.05

        return throttle, yaw
    
    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            debug_frame = frame.copy()
            
            # Seguir línea
            throttle, yaw = self.follow_line(frame, drawing_frame=debug_frame)
            
            # Crear y publicar comando de movimiento
            cmd_msg = Twist()
            cmd_msg.linear.x = float(throttle)
            cmd_msg.angular.z = float(yaw)
            self.cmd_pub.publish(cmd_msg)
            
            # Agregar información de debug
            cv2.putText(debug_frame, "LINE FOLLOWER", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(debug_frame, f"Throttle: {throttle:.3f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(debug_frame, f"Yaw: {math.degrees(yaw):.1f}°", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(debug_frame, f"Authority: {self.authority:.2f}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(debug_frame, f"Straight count: {self.straight_counter}", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Publicar authority actual
            auth_msg = Float32()
            auth_msg.data = self.authority
            self.authority_pub.publish(auth_msg)
            
            # Publicar imagen de debug
            debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
            self.debug_pub.publish(debug_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error in line following: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerNode()
    
    try:
        node.get_logger().info("🚀 Line Follower Node started")
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Stopping Line Follower Node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()