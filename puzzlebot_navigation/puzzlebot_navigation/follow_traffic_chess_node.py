#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from simple_pid import PID

# ======= BUZZER IMPORTS (LAPTOP - SOLO COMANDOS) =======
import threading
import time

class LineFollowerNode(Node):
    def __init__(self):
        super().__init__('line_follower_node')

        # ======= BUZZER MELODIES (VERSIÓN ORIGINAL QUE FUNCIONA) =======
        self.melodies = {
            "super_mario_level_complete": [
                (659, 150),  # E5
                (784, 150),  # G5
                (1319, 150), # E6
                (1047, 150), # C6
                (1175, 150), # D6
                (1568, 300), # G6
            ],
            "zelda_secret_unlocked": [
                (392, 150),  # G4
                (523, 150),  # C5
                (659, 150),  # E5
                (784, 150),  # G5
                (1047, 300), # C6
            ],
            "custom_success_chime": [
                (440, 200),  # A4
                (523, 200),  # C5
                (659, 200),  # E5
                (880, 400),  # A5
                (659, 200),  # E5
                (880, 800),  # A5
            ],
            "mission_complete": [
                (523, 200),  # C5
                (659, 200),  # E5
                (784, 200),  # G5
                (1047, 400), # C6
                (784, 200),  # G5
                (1047, 600), # C6
            ],
            # ======= NUEVAS MELODÍAS =======
            "star_wars_victory": [
                (392, 300),  # G4
                (523, 300),  # C5
                (659, 300),  # E5
                (784, 600),  # G5
            ],
            "windows_xp_logon": [
                (587, 200),  # D5
                (784, 200),  # G5
                (740, 200),  # F#5
                (880, 400),  # A5
            ],
            "tetris_line_clear": [
                (659, 150),  # E5
                (523, 150),  # C5
                (587, 150),  # D5
                (659, 150),  # E5
                (587, 150),  # D5
                (523, 150),  # C5
                (440, 300),  # A4
            ],
            "pokemon_victory": [
                (523, 200),  # C5
                (659, 200),  # E5
                (784, 200),  # G5
                (523, 200),  # C5
                (659, 200),  # E5
                (784, 400),  # G5
                (1047, 600), # C6
            ],
            "beep_simple": [
                (800, 200),  # Beep simple
                (0, 100),    # Silencio
                (800, 200),  # Beep simple
            ]
        }
        
        # Adjust speed
        speed = 1.5
        for melody in self.melodies.values():
            for i in range(len(melody)):
                freq, duration = melody[i]
                melody[i] = (freq, duration / speed)

        self.bridge = CvBridge()
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_safe', 10)
        self.image_sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.debug_pub = self.create_publisher(Image, '/debug_image', 10)
        
        # Publishers para debug del semáforo
        self.semaforo_pub = self.create_publisher(Image, '/semaforo_debug', 10)
        self.mask_pub = self.create_publisher(Image, '/mask_debug', 10)
        
        # ======= NUEVO: PUBLISHERS PARA TABLERO DE AJEDREZ =======
        self.chessboard_pub = self.create_publisher(Image, '/debug_chessboard', 10)
        self.flag_pub = self.create_publisher(Bool, '/flag_close', 10)
        
        # ======= PUBLISHER PARA BUZZER REMOTO (JETSON) =======
        from std_msgs.msg import String
        self.melody_pub = self.create_publisher(String, '/play_melody', 10)

        self.max_yaw = math.radians(60)
        self.max_thr = 0.2
        self.align_thres = 0.3
        self.yaw_pid = PID(Kp=0.8, Ki=0.0, Kd=0.15, setpoint=0.0, output_limits=(-self.max_yaw, self.max_yaw))

        # Fallback control
        self.last_yaw = 0.0
        self.last_thr = 0.05
        
        # Para corregir sesgo al salir de curvas
        self.last_angle = 0.0
        self.transition_frames = 0
        self.straight_counter = 0

        # ======= VARIABLES PARA SEMÁFORO =======
        self.semaforo_active = True  # True = verde/normal, False = rojo/parado
        self.slow_mode = False       # True = amarillo detectado

        # ======= NUEVAS VARIABLES PARA TABLERO DE AJEDREZ =======
        # Parámetros de cámara
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
        self.chessboard_detected = False
        self.chessboard_close = False
        self.mission_completed = False  # 🏁 NUEVA: Estado de misión completada

        self.get_logger().info("Line follower node with semaforo and chessboard detection initialized")

    # ======= BUZZER FUNCTIONS (LAPTOP - ENVÍA COMANDOS) =======
    def play_melody_nonblocking(self, melody_name):
        """Envía comando de melodía a la Jetson via ROS2"""
        try:
            from std_msgs.msg import String
            msg = String()
            msg.data = melody_name
            self.melody_pub.publish(msg)
            self.get_logger().info(f"🎵 Comando enviado a Jetson: {melody_name}")
        except Exception as e:
            self.get_logger().error(f"❌ Error enviando comando: {e}")
        
        # Simulación local para debug
        self.simulate_locally(melody_name)
    
    def simulate_locally(self, melody_name):
        """Simula localmente para ver en logs"""
        def simulate():
            self.get_logger().info(f"🎵 [LAPTOP SIMULATION] {melody_name}")
            melody = self.melodies.get(melody_name, [])
            for i, (freq, duration) in enumerate(melody):
                note_name = self.freq_to_note(freq)
                print(f"  🎶 Nota {i+1}: {note_name} ({freq}Hz)")
                time.sleep(duration / 1000.0)
            self.get_logger().info(f"✅ [LAPTOP SIMULATION] {melody_name} completada")
        
        t = threading.Thread(target=simulate)
        t.daemon = True
        t.start()
    
    def freq_to_note(self, freq):
        """Convierte frecuencia a nota musical"""
        notes = {
            392: "Sol4", 440: "La4", 523: "Do5", 587: "Re5", 659: "Mi5", 
            784: "Sol5", 880: "La5", 1047: "Do6", 1175: "Re6", 1319: "Mi6", 1568: "Sol6"
        }
        return notes.get(freq, f"{freq}Hz")

    # ======= NUEVAS FUNCIONES PARA TABLERO DE AJEDREZ =======
    def undistort_frame(self, frame):
        """Corrige la distorsión de la cámara"""
        return cv2.undistort(frame, self.K, self.D)

    def get_checkerboard_corners(self, frame):
        """Detecta las esquinas del tablero de ajedrez"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
        return corners if ret else None

    def estimate_distance_from_height(self, corners):
        """Estima la distancia basándose en la altura aparente del patrón"""
        ys = corners[:, :, 1].flatten()
        h_pix = ys.max() - ys.min()
        H_real = self.square_size * (self.pattern_size[1] - 1)
        Z = (self.f_y * H_real) / h_pix
        return Z, h_pix

    def detect_chessboard(self, frame):
        """
        Detecta tablero de ajedrez y actualiza estados.
        Retorna el frame con visualizaciones del tablero.
        """
        chessboard_debug = frame.copy()
        
        # Corregir distorsión
        undist = self.undistort_frame(frame)
        
        # Detectar esquinas
        corners = self.get_checkerboard_corners(undist)
        
        self.chessboard_detected = False
        self.chessboard_close = False
        
        if corners is not None:
            self.chessboard_detected = True
            
            # Calcular distancia
            Z, h_pix = self.estimate_distance_from_height(corners)
            self.chessboard_close = Z <= self.threshold
            
            # 🏁 NUEVA LÓGICA: Si tablero está cerca, completar misión
            if self.chessboard_close and not self.mission_completed:
                self.mission_completed = True
                # 🎵 REPRODUCIR MELODÍA DE VICTORIA
                self.play_melody_nonblocking("mission_complete")
            
            # Dibujar esquinas detectadas
            cv2.drawChessboardCorners(chessboard_debug, self.pattern_size, corners, True)
            
            # Información de debug
            if self.mission_completed:
                status_text = f"🏁 MISIÓN COMPLETADA! (Z={Z:.2f}m)"
                color = (0, 255, 0)
            else:
                status_text = f"TABLERO CERCA (Z={Z:.2f}m)" if self.chessboard_close else f"TABLERO LEJOS (Z={Z:.2f}m)"
                color = (0, 255, 0) if self.chessboard_close else (255, 0, 0)
            
            cv2.putText(chessboard_debug, status_text, (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(chessboard_debug, f"Altura: {h_pix:.1f} pix", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Log para debugging
            if self.chessboard_close:
                if not self.mission_completed:
                    self.get_logger().info(f'🏁 ¡MISIÓN COMPLETADA! Tablero detectado a {Z:.3f}m - ROBOT DETENIDO PERMANENTEMENTE')
                    self.mission_completed = True
                else:
                    self.get_logger().info(f'♟️  Tablero cerca: {Z:.3f}m - Robot ya detenido')
            else:
                self.get_logger().info(f'♟️  Tablero detectado: {Z:.3f}m')
        else:
            cv2.putText(chessboard_debug, "NO TABLERO DETECTADO", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return chessboard_debug

    # ======= FUNCIONES ORIGINALES DEL LINE FOLLOWER (SIN CAMBIOS) =======
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
            
            # Compensación agresiva e inmediata
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

    def is_circle(self, contour):
        """Detecta círculos extremadamente pequeños y lejanos para semáforos"""
        area = cv2.contourArea(contour)
        if area < 15:
            return False
            
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        if area > 6000:
            if aspect_ratio > 3.0 or aspect_ratio < 0.3:
                return False
            if circularity < 0.4:
                return False
        
        is_circular = (
            circularity > 0.25 and
            0.4 < aspect_ratio < 2.5 and
            15 < area < 20000
        )
        
        return is_circular

    def detect_semaforo(self, frame):
        """Detecta colores de semáforo y actualiza los estados del robot"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Rangos de colores HSV
        lower_green = np.array([35, 40, 30])
        upper_green = np.array([90, 255, 255])

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

        # Filtro morfológico para verde
        kernel = np.ones((5, 5), np.uint8)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)

        self.mask_pub.publish(self.bridge.cv2_to_imgmsg(mask_green, encoding='mono8'))

        # Encontrar contornos
        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Reset de estados
        self.slow_mode = False
        
        semaforo_debug = frame.copy()

        # Amarillo detectado → modo lento
        if contours_yellow:
            largest_yellow = max(contours_yellow, key=cv2.contourArea)
            area_yellow = cv2.contourArea(largest_yellow)
            
            if area_yellow > 500:
                is_circle = self.is_circle(largest_yellow)
                x, y, w, h = cv2.boundingRect(largest_yellow)
                y_center = y + h // 2
                frame_height = frame.shape[0]
                is_ground_object = y_center > frame_height * 0.75
                
                if is_circle and not is_ground_object:
                    perimeter = cv2.arcLength(largest_yellow, True)
                    circularity = 4 * np.pi * area_yellow / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    self.slow_mode = True
                    self.get_logger().info(f'🟡 CÍRCULO Amarillo detectado: MODO LENTO | Circularidad: {circularity:.3f}')
                    # 🎵 Sonido para amarillo (opcional)
                    # self.play_melody_nonblocking("zelda_secret_unlocked")
                    cv2.drawContours(semaforo_debug, [largest_yellow], -1, (0, 255, 255), 3)
                    cv2.putText(semaforo_debug, "AMARILLO CIRCULAR - LENTO", (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Rojo detectado → detener
        if contours_red:
            largest_red = max(contours_red, key=cv2.contourArea)
            area_red = cv2.contourArea(largest_red)
            
            if area_red > 15:
                is_circle = self.is_circle(largest_red)
                x, y, w, h = cv2.boundingRect(largest_red)
                y_center = y + h // 2
                frame_height = frame.shape[0]
                is_ground_object = y_center > frame_height * 0.90
                
                if is_circle and not is_ground_object:
                    self.get_logger().info(f'🟥 CÍRCULO Rojo detectado: DETENIENDO | Área: {area_red:.0f}')
                    self.semaforo_active = False
                    cv2.drawContours(semaforo_debug, [largest_red], -1, (0, 0, 255), 3)
                    cv2.putText(semaforo_debug, "ROJO CIRCULAR - DETENIDO", (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    return semaforo_debug

        # Verde detectado → activar
        if contours_green:
            largest_green = max(contours_green, key=cv2.contourArea)
            area_green = cv2.contourArea(largest_green)
            
            if area_green > 15:
                is_circle = self.is_circle(largest_green)
                x, y, w, h = cv2.boundingRect(largest_green)
                y_center = y + h // 2
                frame_height = frame.shape[0]
                is_ground_object = y_center > frame_height * 0.90
                
                if is_circle and not is_ground_object:
                    perimeter = cv2.arcLength(largest_green, True)
                    circularity = 4 * np.pi * area_green / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    self.get_logger().info(f'🟢 CÍRCULO Verde | Área: {area_green:.2f} - ACTIVO')
                    self.semaforo_active = True
                    cv2.drawContours(semaforo_debug, [largest_green], -1, (0, 255, 0), 3)
                    cv2.putText(semaforo_debug, "VERDE CIRCULAR - ACTIVO", (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Mostrar estado actual
        status_color = (0, 255, 0) if self.semaforo_active else (0, 0, 255)
        status_text = "ACTIVO" if self.semaforo_active else "DETENIDO"
        cv2.putText(semaforo_debug, f"Line Follower: {status_text}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        if self.slow_mode:
            cv2.putText(semaforo_debug, "Modo Lento Activado", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return semaforo_debug

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # ======= PROCESAR SEMÁFORO =======
        semaforo_debug = self.detect_semaforo(frame)
        semaforo_msg = self.bridge.cv2_to_imgmsg(semaforo_debug, encoding='bgr8')
        self.semaforo_pub.publish(semaforo_msg)
        
        # ======= PROCESAR TABLERO DE AJEDREZ =======
        chessboard_debug = self.detect_chessboard(frame)
        chessboard_msg = self.bridge.cv2_to_imgmsg(chessboard_debug, encoding='bgr8')
        self.chessboard_pub.publish(chessboard_msg)
        
        # Publicar estado de proximidad del tablero
        self.flag_pub.publish(Bool(data=bool(self.chessboard_close)))
        
        # ======= LÓGICA DEL LINE FOLLOWER =======
        debug = frame.copy()
        
        # 🏁 NUEVA LÓGICA: Si misión completada, NO HACER NADA MÁS
        if self.mission_completed:
            v, w = 0.0, 0.0
            cv2.putText(debug, "MISSION COMPLETED - ROBOT STOPPED", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(debug, "CHESSBOARD REACHED - FINAL STOP", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Solo procesar line following si el semáforo está activo Y la misión NO está completada
        elif self.semaforo_active:
            v, w = self.follow_line(frame, drawing_frame=debug)
            
            # Aplicar modo lento si detectamos amarillo
            if self.slow_mode:
                v *= 0.5
                w *= 0.5
                cv2.putText(debug, "SLOW MODE", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                           
        else:
            # Semáforo en rojo - detener completamente
            v, w = 0.0, 0.0
            cv2.putText(debug, "STOPPED BY SEMAFORO", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # ======= MOSTRAR ESTADO DEL TABLERO EN DEBUG PRINCIPAL =======
        if self.mission_completed:
            cv2.putText(debug, "🏁 FINAL DESTINATION REACHED", (10, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif self.chessboard_detected:
            chessboard_status = "TABLERO CERCA" if self.chessboard_close else "TABLERO LEJOS"
            chessboard_color = (0, 255, 255) if self.chessboard_close else (255, 255, 0)
            cv2.putText(debug, chessboard_status, (10, 170), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, chessboard_color, 2)

        # Publicar comandos de velocidad
        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.cmd_pub.publish(twist)

        # Publicar debug del line follower
        debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
        self.debug_pub.publish(debug_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Nodo interrumpido por usuario")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
