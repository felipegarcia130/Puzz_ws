import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import Bool, Int32, String
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import time
from typing import List, Tuple, Optional
from enum import Enum

class RobotState(Enum):
    FOLLOWING_LINE = 1
    INTERSECTION_DETECTED = 2
    TURNING = 3
    LOST = 4

class TurnDirection(Enum):
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3

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

class IntersectionDetector:
    """Detector de intersecciones integrado"""
    def __init__(self):
        self.min_line_length = 30
        self.max_line_gap = 10
        self.hough_threshold = 50
        self.intersection_threshold = 3
        
    def detect_intersection(self, image) -> Tuple[Optional[Tuple[int, int]], int, np.ndarray]:
        """
        Detecta intersecciones en la imagen
        Retorna: (punto_intersección, número_de_caminos, imagen_debug)
        """
        height, width = image.shape[:2]
        debug_image = image.copy()
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aplicar filtro Gaussiano para reducir ruido
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Binarización adaptativa para detectar líneas negras
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morfología para limpiar la imagen
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Detectar bordes
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        
        # Detectar líneas usando transformada de Hough
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=self.hough_threshold,
                               minLineLength=self.min_line_length, maxLineGap=self.max_line_gap)
        
        if lines is None:
            return None, 0, debug_image
        
        # Clasificar líneas por orientación
        horizontal_lines = []
        vertical_lines = []
        diagonal_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calcular ángulo
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angle = abs(angle)
            
            if angle < 20 or angle > 160:  # Líneas horizontales
                horizontal_lines.append(line[0])
            elif 70 < angle < 110:  # Líneas verticales
                vertical_lines.append(line[0])
            else:  # Líneas diagonales
                diagonal_lines.append(line[0])
        
        # Dibujar líneas en imagen de debug
        for line in horizontal_lines:
            cv2.line(debug_image, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)
        for line in vertical_lines:
            cv2.line(debug_image, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 2)
        for line in diagonal_lines:
            cv2.line(debug_image, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)
        
        # Encontrar intersecciones
        intersections = self._find_intersections(horizontal_lines, vertical_lines, diagonal_lines)
        
        if not intersections:
            return None, 0, debug_image
        
        # Filtrar intersecciones por proximidad al centro de la imagen
        center_x, center_y = width // 2, height // 2
        closest_intersection = min(intersections, 
                                 key=lambda p: (p[0] - center_x)**2 + (p[1] - center_y)**2)
        
        # Contar caminos disponibles en la intersección
        num_paths = self._count_available_paths(closest_intersection, horizontal_lines, 
                                             vertical_lines, diagonal_lines)
        
        # Dibujar intersección en imagen de debug
        cv2.circle(debug_image, closest_intersection, 10, (255, 255, 0), -1)
        cv2.putText(debug_image, f'Paths: {num_paths}', 
                   (closest_intersection[0] + 15, closest_intersection[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Solo considerar como intersección si hay suficientes caminos
        if num_paths >= self.intersection_threshold:
            return closest_intersection, num_paths, debug_image
        else:
            return None, num_paths, debug_image
    
    def _find_intersections(self, h_lines, v_lines, d_lines):
        """Encuentra puntos de intersección entre líneas"""
        intersections = []
        all_lines = h_lines + v_lines + d_lines
        
        for i in range(len(all_lines)):
            for j in range(i + 1, len(all_lines)):
                intersection = self._line_intersection(all_lines[i], all_lines[j])
                if intersection:
                    intersections.append(intersection)
        
        # Eliminar intersecciones duplicadas (muy cercanas)
        filtered_intersections = []
        for point in intersections:
            is_duplicate = False
            for existing in filtered_intersections:
                if np.sqrt((point[0] - existing[0])**2 + (point[1] - existing[1])**2) < 20:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered_intersections.append(point)
        
        return filtered_intersections
    
    def _line_intersection(self, line1, line2):
        """Calcula la intersección entre dos líneas"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (int(x), int(y))
        
        return None
    
    def _count_available_paths(self, intersection, h_lines, v_lines, d_lines):
        """Cuenta el número de caminos disponibles desde una intersección"""
        x, y = intersection
        radius = 30  # Radio para buscar líneas cercanas
        
        paths = 0
        
        # Verificar líneas horizontales (izquierda y derecha)
        for line in h_lines:
            if self._point_near_line((x, y), line, radius):
                paths += 1
                break
        
        # Verificar líneas verticales (arriba y abajo)
        for line in v_lines:
            if self._point_near_line((x, y), line, radius):
                paths += 1
                break
        
        # Verificar líneas diagonales
        diagonal_count = 0
        for line in d_lines:
            if self._point_near_line((x, y), line, radius):
                diagonal_count += 1
        
        paths += min(diagonal_count, 2)  # Máximo 2 diagonales
        
        return paths
    
    def _point_near_line(self, point, line, threshold):
        """Verifica si un punto está cerca de una línea"""
        x0, y0 = point
        x1, y1, x2, y2 = line
        
        # Distancia de punto a línea
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        
        distance = abs(A * x0 + B * y0 + C) / np.sqrt(A**2 + B**2)
        
        return distance < threshold

class LineFollowerNode(Node):
    def __init__(self):
        super().__init__('line_follower_node')

        # Configuración de parámetros
        self.declare_parameter('intersection_detection', True)
        self.declare_parameter('decision_strategy', 'straight_priority')  # straight_priority, left_priority, right_priority
        self.declare_parameter('intersection_pause_time', 1.0)  # Tiempo de pausa en intersección
        self.declare_parameter('debug_mode', True)
        
        self.intersection_detection_enabled = self.get_parameter('intersection_detection').value
        self.decision_strategy = self.get_parameter('decision_strategy').value
        self.intersection_pause_time = self.get_parameter('intersection_pause_time').value
        self.debug_mode = self.get_parameter('debug_mode').value

        self.bridge = CvBridge()
        
        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_safe', 10)
        self.debug_pub = self.create_publisher(Image, '/debug_image', 10)
        self.intersection_pub = self.create_publisher(Point, '/intersection_point', 10)
        self.intersection_detected_pub = self.create_publisher(Bool, '/intersection_detected', 10)
        self.num_paths_pub = self.create_publisher(Int32, '/num_available_paths', 10)
        self.robot_state_pub = self.create_publisher(String, '/robot_state', 10)
        
        # Subscribers
        self.image_sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)

        # Control parameters
        self.max_yaw = math.radians(60)
        self.max_thr = 0.2
        self.align_thres = 0.3
        self.yaw_pid = SimplePID(Kp=0.8, Ki=0.0, Kd=0.15, setpoint=0.0, output_limits=(-self.max_yaw, self.max_yaw))

        # Fallback control
        self.last_yaw = 0.0
        self.last_thr = 0.05
        
        # Para corregir sesgo al salir de curvas
        self.last_angle = 0.0
        self.transition_frames = 0
        self.straight_counter = 0

        # Estado del robot y detector de intersecciones
        self.robot_state = RobotState.FOLLOWING_LINE
        self.intersection_detector = IntersectionDetector()
        self.intersection_timer = None
        self.current_intersection_point = None
        self.current_num_paths = 0
        self.turn_direction = TurnDirection.STRAIGHT
        self.intersection_pause_start = None
        
        # Historial para evitar oscilaciones en intersecciones
        self.intersection_detected_frames = 0
        self.min_intersection_frames = 5  # Mínimo de frames consecutivos para confirmar intersección

        self.get_logger().info("Line follower with intersection detection initialized")

    def adaptive_thres(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 141, 6)
        return mask

    def get_line_mask(self, frame):
        mask = self.adaptive_thres(frame)
        mask[:int(frame.shape[0] * (1 - 0.4)), :] = 0
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

    def decide_turn_direction(self, num_paths: int) -> TurnDirection:
        """Decide la dirección a tomar en una intersección"""
        if self.decision_strategy == 'straight_priority':
            return TurnDirection.STRAIGHT
        elif self.decision_strategy == 'left_priority':
            return TurnDirection.LEFT
        elif self.decision_strategy == 'right_priority':
            return TurnDirection.RIGHT
        else:
            # Estrategia aleatoria o por defecto
            return TurnDirection.STRAIGHT

    def execute_turn(self, direction: TurnDirection) -> Tuple[float, float]:
        """Ejecuta un giro específico"""
        if direction == TurnDirection.LEFT:
            return 0.1, math.radians(30)  # Velocidad lenta, giro izquierda
        elif direction == TurnDirection.RIGHT:
            return 0.1, math.radians(-30)  # Velocidad lenta, giro derecha
        else:  # STRAIGHT
            return 0.15, 0.0  # Seguir recto

    def follow_line(self, frame, drawing_frame=None):
        throttle, yaw = 0.0, 0.0
        frame_height, frame_width = frame.shape[:2]

        # Detectar intersecciones si está habilitado
        intersection_point = None
        num_paths = 0
        if self.intersection_detection_enabled:
            intersection_point, num_paths, intersection_debug = self.intersection_detector.detect_intersection(frame)
            
            # Superponer información de intersección en el frame de debug
            if drawing_frame is not None and intersection_debug is not None:
                # Combinar las imágenes de debug
                alpha = 0.7
                cv2.addWeighted(drawing_frame, alpha, intersection_debug, 1-alpha, 0, drawing_frame)

        # Lógica de estado del robot
        if intersection_point and self.robot_state == RobotState.FOLLOWING_LINE:
            self.intersection_detected_frames += 1
            if self.intersection_detected_frames >= self.min_intersection_frames:
                self.robot_state = RobotState.INTERSECTION_DETECTED
                self.current_intersection_point = intersection_point
                self.current_num_paths = num_paths
                self.turn_direction = self.decide_turn_direction(num_paths)
                self.intersection_pause_start = time.time()
                self.get_logger().info(f'Intersección detectada con {num_paths} caminos. Decisión: {self.turn_direction.name}')
        elif not intersection_point:
            self.intersection_detected_frames = 0
            if self.robot_state == RobotState.INTERSECTION_DETECTED:
                self.robot_state = RobotState.FOLLOWING_LINE

        # Publicar información de intersección
        if intersection_point:
            point_msg = Point()
            point_msg.x = float(intersection_point[0])
            point_msg.y = float(intersection_point[1])
            point_msg.z = 0.0
            self.intersection_pub.publish(point_msg)

        bool_msg = Bool()
        bool_msg.data = (intersection_point is not None)
        self.intersection_detected_pub.publish(bool_msg)

        paths_msg = Int32()
        paths_msg.data = num_paths
        self.num_paths_pub.publish(paths_msg)

        state_msg = String()
        state_msg.data = self.robot_state.name
        self.robot_state_pub.publish(state_msg)

        # Comportamiento según el estado
        if self.robot_state == RobotState.INTERSECTION_DETECTED:
            # Pausa en la intersección antes de decidir
            if time.time() - self.intersection_pause_start < self.intersection_pause_time:
                throttle, yaw = 0.05, 0.0  # Moverse muy lentamente
                if drawing_frame is not None:
                    cv2.putText(drawing_frame, f"INTERSECTION: {self.turn_direction.name}", 
                               (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            else:
                # Ejecutar el giro decidido
                self.robot_state = RobotState.TURNING
                
        elif self.robot_state == RobotState.TURNING:
            # Ejecutar giro por un tiempo limitado
            throttle, yaw = self.execute_turn(self.turn_direction)
            if drawing_frame is not None:
                cv2.putText(drawing_frame, f"TURNING: {self.turn_direction.name}", 
                           (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Volver al seguimiento normal después de un tiempo
            if time.time() - self.intersection_pause_start > self.intersection_pause_time + 2.0:
                self.robot_state = RobotState.FOLLOWING_LINE
                self.yaw_pid.reset()  # Reset PID para estabilizar
                
        else:  # FOLLOWING_LINE
            # Seguimiento normal de línea (código original)
            line = self.get_middle_line(frame, drawing_frame)

            if line:
                contour, (pt1, pt2, angle, cx, cy, length) = line
                x, y, w, h = cv2.boundingRect(contour)
                
                # Corrección para sesgo (código original)
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
                
                if is_straight:
                    # Método mejorado para rectas
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
                
                # Correcciones de sesgo
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
                        cv2.putText(drawing_frame, f"STRONG CORRECTION: {strength:.3f}", (10, 150), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                
                elif is_straight and self.straight_counter > 2:
                    continuous_correction = -0.04
                    normalized_x += continuous_correction
                    
                    if drawing_frame is not None:
                        cv2.putText(drawing_frame, "CONTINUOUS CORRECTION", (10, 150), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                if is_straight and abs(normalized_x) < 0.015:
                    normalized_x = 0.0
                
                yaw = self.yaw_pid(normalized_x)
                self.last_angle = angle

                alignment = 1 - abs(normalized_x)
                x = ((alignment - self.align_thres) / (1 - self.align_thres))
                throttle = self.max_thr * x

                self.last_yaw = yaw
                self.last_thr = throttle

                if drawing_frame is not None:
                    cv2.line(drawing_frame, (center_x, 0), (center_x, frame_height), (255, 0, 0), 2)
                    cv2.line(drawing_frame, (frame_width//2, 0), (frame_width//2, frame_height), (0, 255, 255), 1)
                    cv2.putText(drawing_frame, f"v: {throttle:.2f} m/s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(drawing_frame, f"w: {math.degrees(yaw):.2f} deg/s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(drawing_frame, f"straight_cnt: {self.straight_counter}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                yaw = self.last_yaw
                throttle = 0.05
                if drawing_frame is not None:
                    cv2.putText(drawing_frame, "Fallback mode (no line)", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return throttle, yaw

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        debug = frame.copy()
        v, w = self.follow_line(frame, drawing_frame=debug)

        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.cmd_pub.publish(twist)

        if self.debug_mode:
            debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
            self.debug_pub.publish(debug_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()