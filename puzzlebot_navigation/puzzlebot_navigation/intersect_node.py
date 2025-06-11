import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import time

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
        #Reinicia el controlador PID
        self._integral = 0.0
        self._last_error = 0.0
        self._last_time = None

# ==================== FUNCIONES PARA DETECCIÓN DE INTERSECCIONES ====================

def find_dots_for_intersection(frame, drawing_frame=None):
    
    #Detecta puntos que pueden formar líneas punteadas de intersección
    
    if drawing_frame is None:
        drawing_frame = frame.copy()
    
    # Umbralización adaptativa para crear máscara binaria
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 5)

    # Encontrar contornos en la máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 20]
    dots = []

    # Relación de aspecto máxima permitida
    max_aspect_ratio = 10.0

    for cnt in contours:
        # Aproximar el contorno a un polígono
        epsilon = 0.03 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Verificar si es un cuadrilátero convexo
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            # Filtrar formas muy alargadas
            if min(w, h) == 0 or max(w, h) / min(w, h) > max_aspect_ratio:
                continue
            center = (x + w // 2, y + h // 2)
            dots.append(center)
            
            # Dibujar puntos detectados
            if drawing_frame is not None:
                cv2.circle(drawing_frame, center, 3, (0, 0, 255), -1)

    return dots

def cluster_collinear_points(dots, min_points=4, threshold=8, outlier_factor=8.0):
    
    #Agrupa puntos que son aproximadamente colineales

    remaining = dots.copy()
    groups = []

    while len(remaining) >= min_points:
        best_group = []
        
        # Probar cada par como línea candidata
        for i in range(len(remaining)):
            for j in range(i+1, len(remaining)):
                p1 = remaining[i]
                p2 = remaining[j]
                
                # Calcular parámetros de línea
                if p2[0] - p1[0] == 0:
                    a, b, c = 1, 0, -p1[0]
                    direction = (0, 1)
                else:
                    slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
                    a, b, c = -slope, 1, -((-slope) * p1[0] + 1 * p1[1])
                    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                    L = math.hypot(dx, dy)
                    direction = (dx / L, dy / L)

                # Encontrar puntos cerca de esta línea
                candidate_group = []
                for p in remaining:
                    dist = abs(a * p[0] + b * p[1] + c) / math.sqrt(a**2 + b**2)
                    if dist < threshold:
                        candidate_group.append(p)

                # Filtrar outliers
                if len(candidate_group) >= 2:
                    projections = [((pt[0] * direction[0] + pt[1] * direction[1]), pt)
                                 for pt in candidate_group]
                    projections.sort(key=lambda x: x[0])
                    
                    proj_values = [proj for proj, _ in projections]
                    if len(proj_values) > 1:
                        avg_gap = (proj_values[-1] - proj_values[0]) / (len(proj_values) - 1)
                    else:
                        avg_gap = 0

                    filtered = []
                    for idx, (proj, pt) in enumerate(projections):
                        if idx == 0:
                            gap = proj_values[1] - proj if len(proj_values) > 1 else 0
                        elif idx == len(proj_values) - 1:
                            gap = proj - proj_values[idx - 1]
                        else:
                            gap = min(proj - proj_values[idx - 1], proj_values[idx + 1] - proj)
                        
                        if gap <= outlier_factor * avg_gap or avg_gap == 0:
                            filtered.append(pt)
                    candidate_group = filtered

                if len(candidate_group) > len(best_group):
                    best_group = candidate_group

        if len(best_group) >= min_points:
            groups.append(best_group)
            remaining = [p for p in remaining if p not in best_group]
        else:
            break

    return groups

def find_line_endpoints(points):
    #Encuentra los extremos de una línea formada por puntos
    if len(points) < 2:
        return None, None
    
    max_distance = 0
    endpoint1, endpoint2 = None, None
    
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            d = math.hypot(points[j][0] - points[i][0], points[j][1] - points[i][1])
            if d > max_distance:
                max_distance = d
                endpoint1, endpoint2 = points[i], points[j]
    
    return endpoint1, endpoint2

def get_dotted_lines(frame, drawing_frame=None):

    #Detecta líneas punteadas agrupando puntos colineales
    
    if drawing_frame is None:
        drawing_frame = frame.copy()
        
    dots = find_dots_for_intersection(frame, drawing_frame)
    groups = cluster_collinear_points(dots)
    dotted_lines = []
    
    for group in groups:
        endpoints = find_line_endpoints(group)
        if endpoints[0] is not None and endpoints[1] is not None:
            dotted_lines.append(endpoints)

    # Dibujar líneas detectadas
    if drawing_frame is not None:
        for line in dotted_lines:
            cv2.line(drawing_frame, line[0], line[1], (255, 0, 0), 2)

    return dotted_lines

def identify_intersection(frame, drawing_frame=None):
    
    #Identifica una intersección y clasifica las direcciones
    #Retorna: [back, left, right, front] donde cada elemento es None o (line, center, angle)
    
    if drawing_frame is None:
        drawing_frame = frame.copy()

    dotted_lines = get_dotted_lines(frame, drawing_frame)
    
    if not dotted_lines:
        return [None, None, None, None]
    
    # Calcular centros y ángulos de las líneas
    centers = [((l[0][0] + l[1][0]) // 2, (l[0][1] + l[1][1]) // 2) for l in dotted_lines]
    angles = [((math.degrees(math.atan2(l[1][1] - l[0][1], l[1][0] - l[0][0])) + 90) % 180) - 90 for l in dotted_lines]
    dotted_lines_info = list(zip(dotted_lines, centers, angles))
    
    # Clasificar líneas por orientación
    vert_threshold = 30
    verticals = [dl for dl in dotted_lines_info if abs(dl[2]) > vert_threshold]
    horizontals = [dl for dl in dotted_lines_info if abs(dl[2]) <= vert_threshold]

    # Identificar direcciones
    frame_height, frame_width = frame.shape[:2]
    mid_x = frame_width / 2
    
    # Líneas horizontales (back/front)
    horizontal_candidates = [h for h in horizontals if h[1][1] / frame_height >= 0.3]
    horizontal_sorted = sorted(horizontal_candidates, key=lambda x: x[1][1], reverse=True)
    
    back = horizontal_sorted[0] if horizontal_sorted else None
    front = horizontal_sorted[1] if len(horizontal_sorted) > 1 else None
    
    # Líneas verticales (left/right)
    left_candidates = [v for v in verticals if v[1][0] < mid_x]
    left = sorted(left_candidates, key=lambda x: x[1][0], reverse=True)[0] if left_candidates else None
    
    right_candidates = [v for v in verticals if v[1][0] > mid_x]
    right = sorted(right_candidates, key=lambda x: x[1][0])[0] if right_candidates else None
    
    # Verificar si back/front están en orden correcto
    all_sorted = sorted(dotted_lines_info, key=lambda x: x[1][1], reverse=True)
    if back and all_sorted and back != all_sorted[0]:
        front = back
        back = None

    directions = [back, left, right, front]
    
    # Dibujar indicadores de dirección
    if drawing_frame is not None:
        direction_names = ['BACK', 'LEFT', 'RIGHT', 'FRONT']
        colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0)]
        
        for i, (name, direction) in enumerate(zip(direction_names, directions)):
            color = colors[i] if direction is not None else (128, 128, 128)
            cv2.putText(drawing_frame, f"{name}: {'YES' if direction else 'NO'}", 
                       (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Dibujar triángulos para direcciones detectadas
        for dl in filter(None, directions):
            line, center, angle = dl
            angle_rad = math.radians(angle * 1.5 + (90 if dl == back else -90 if dl == front else 0))
            h = 30
            pt1 = (int(center[0] + h * math.cos(angle_rad + 0.3)), 
                   int(center[1] + h * math.sin(angle_rad + 0.3)))
            pt2 = (int(center[0] + h * math.cos(angle_rad - 0.3)), 
                   int(center[1] + h * math.sin(angle_rad - 0.3)))
            cv2.fillPoly(drawing_frame, [np.array([center, pt1, pt2], np.int32)], (0, 255, 0))
    
    return directions

class LineFollowerWithIntersectionNode(Node):
    def __init__(self):
        super().__init__('line_follower_intersection_node')

        self.bridge = CvBridge()
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_safe', 10)
        self.image_sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.debug_pub = self.create_publisher(Image, '/debug_image', 10)

        # Parámetros del seguidor de líneas (mantener los que funcionan)
        self.max_yaw = math.radians(60)
        self.max_thr = 0.2
        self.align_thres = 0.3
        self.yaw_pid = SimplePID(Kp=0.8, Ki=0.0, Kd=0.15, setpoint=0.0, output_limits=(-self.max_yaw, self.max_yaw))

        # Fallback control
        self.last_yaw = 0.0
        self.last_thr = 0.05
        
        # Para corrección de sesgo
        self.last_angle = 0.0
        self.transition_frames = 0
        self.straight_counter = 0

        # ============ NUEVOS PARÁMETROS PARA INTERSECCIONES ============
        # PIDs para control de intersección
        self.intersection_yaw_pid = SimplePID(Kp=1.5, Ki=0.0, Kd=0.1, setpoint=0.0, 
                                            output_limits=(-math.radians(30), math.radians(30)))
        self.intersection_speed_pid = SimplePID(Kp=0.3, Ki=0.0, Kd=0.05, setpoint=0.7, 
                                              output_limits=(-0.15, 0.15))
        
        # Estados de navegación
        self.intersection_detected = False
        self.intersection_confidence = 0
        self.min_confidence = 3  # Frames consecutivos para confirmar intersección
        self.last_intersection = None
        self.intersection_timeout = 0
        
        # Modo de operación
        self.mode = "FOLLOWING"  # "FOLLOWING" o "INTERSECTION"

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

    def follow_line(self, frame, drawing_frame=None):
        #Función original de seguimiento de líneas (sin cambios)
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

            self.last_yaw = yaw
            self.last_thr = throttle

            if drawing_frame is not None:
                cv2.line(drawing_frame, (center_x, 0), (center_x, frame_height), (255, 0, 0), 2)
                cv2.line(drawing_frame, (frame_width//2, 0), (frame_width//2, frame_height), (0, 255, 255), 1)
        else:
            yaw = self.last_yaw
            throttle = 0.05

        return throttle, yaw

    def stop_at_intersection(self, frame, intersection, drawing_frame=None):
        
        #Control PID para detenerse y alinearse en intersección
        
        back, left, right, front = intersection
        throttle, yaw = 0.0, 0.0
        frame_height, frame_width = frame.shape[:2]
        
        # Usar línea trasera o frontal para alineación
        reference_line = back if back else front
        
        if reference_line:
            line, center, angle = reference_line
            
            # Control de orientación (yaw)
            angle_error = math.radians(angle)
            yaw = self.intersection_yaw_pid(angle_error)
            
            # Control de velocidad basado en distancia y alineación
            yaw_threshold_deg = 5.0
            alignment_factor = 1 - (abs(angle) / yaw_threshold_deg) if abs(angle) < yaw_threshold_deg else 0
            
            # Distancia normalizada (0 = cerca del borde inferior, 1 = cerca del borde superior)
            norm_distance = center[1] / frame_height
            target_distance = 0.7  # Detenerse cuando la línea esté al 70% de la imagen
            
            # Combinar distancia y alineación para velocidad
            measured_value = alignment_factor * norm_distance + (1 - alignment_factor) * target_distance
            throttle = self.intersection_speed_pid(measured_value)
            
            if drawing_frame is not None:
                cv2.putText(drawing_frame, f"Intersection angle: {angle:.1f}°", (10, 220), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(drawing_frame, f"Alignment: {alignment_factor:.2f}", (10, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return throttle, yaw

    def image_callback(self, msg):
        #Callback principal con detección de intersecciones
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            debug = frame.copy()
            
            # ============ DETECCIÓN DE INTERSECCIONES ============
            intersection = identify_intersection(frame, drawing_frame=debug)
            back, left, right, front = intersection
            
            # Contar direcciones válidas
            valid_directions = [d for d in intersection if d is not None]
            intersection_detected = len(valid_directions) >= 2
            
            # Sistema de confianza para evitar falsos positivos
            if intersection_detected:
                self.intersection_confidence = min(self.intersection_confidence + 1, self.min_confidence + 2)
                self.last_intersection = intersection
            else:
                self.intersection_confidence = max(self.intersection_confidence - 1, 0)
            
            # Determinar modo de operación
            if self.intersection_confidence >= self.min_confidence:
                self.mode = "INTERSECTION"
                self.intersection_timeout = 30  # Frames para permanecer en modo intersección
            elif self.intersection_timeout > 0:
                self.mode = "INTERSECTION"
                self.intersection_timeout -= 1
            else:
                self.mode = "FOLLOWING"
                # Reset PIDs cuando cambiamos a modo seguimiento
                if hasattr(self, '_last_mode') and self._last_mode == "INTERSECTION":
                    self.yaw_pid.reset()
            
            # ============ CONTROL SEGÚN EL MODO ============
            if self.mode == "INTERSECTION":
                v, w = self.stop_at_intersection(frame, self.last_intersection, drawing_frame=debug)
                
                # Información de debug para intersección
                cv2.putText(debug, "INTERSECTION MODE", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(debug, f"Confidence: {self.intersection_confidence}/{self.min_confidence}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(debug, f"Directions: {len(valid_directions)}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(debug, f"Timeout: {self.intersection_timeout}", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
            else:  # FOLLOWING mode
                v, w = self.follow_line(frame, drawing_frame=debug)
                
                # Información de debug para seguimiento
                cv2.putText(debug, "LINE FOLLOWING MODE", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(debug, f"v: {v:.2f} m/s", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(debug, f"w: {math.degrees(w):.1f}°/s", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(debug, f"straight_cnt: {self.straight_counter}", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Guardar modo anterior
            self._last_mode = self.mode
            
            # ============ PUBLICAR COMANDOS ============
            twist = Twist()
            twist.linear.x = float(v)
            twist.angular.z = float(w)
            self.cmd_pub.publish(twist)
            
            # Publicar imagen de debug
            debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
            self.debug_pub.publish(debug_msg)
            
            # Log de estado
            if self.mode == "INTERSECTION":
                self.get_logger().info(f"INTERSECTION: v={v:.3f}, w={math.degrees(w):.1f}°/s, "
                                     f"dirs={len(valid_directions)}, conf={self.intersection_confidence}")
            
        except Exception as e:
            self.get_logger().error(f"Error in image processing: {str(e)}")
            # En caso de error, mantener último comando seguro
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerWithIntersectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()




