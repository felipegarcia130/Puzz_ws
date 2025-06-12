import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import time
from collections import deque
from itertools import combinations
from dataclasses import dataclass
from typing import Optional
from enum import Enum
from ultralytics import YOLO

# ==================== YOLO INTEGRATION ====================

class YOLOSignDetector:
    def __init__(self, model_path, confidence_threshold=0.6):
        try:
            self.model = YOLO(model_path)
            self.confidence_threshold = confidence_threshold
            self.get_logger = lambda: print  # Fallback logger
            print(f'✅ YOLO model loaded: {model_path}')
            if hasattr(self.model, 'names'):
                print(f'📋 Available classes: {list(self.model.names.values())}')
        except Exception as e:
            print(f'❌ Error loading YOLO model: {e}')
            self.model = None
    
    def detect_signs(self, frame):
        #Detecta señales con YOLO y retorna información en formato compatible
        if self.model is None:
            return [], [], [], []
        
        try:
            results = self.model.predict(
                source=frame,
                conf=self.confidence_threshold,
                verbose=False
            )
            
            boxes = []
            sign_types = []
            confidences = []
            class_names = []
            
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    # Obtener coordenadas del bounding box
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    boxes.append([x1, y1, x2, y2])
                    
                    # Obtener clase y confianza
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = self.model.names[cls] if cls < len(self.model.names) else f"class_{cls}"
                    
                    # Mapear nombres de clases a tipos de señales
                    sign_type = self.map_class_to_sign_type(class_name)
                    
                    sign_types.append(sign_type)
                    confidences.append(conf)
                    class_names.append(class_name)
            
            return boxes, sign_types, confidences, class_names
            
        except Exception as e:
            print(f'Error in YOLO detection: {e}')
            return [], [], [], []
    
    def map_class_to_sign_type(self, class_name):
        #Mapea nombres de clases YOLO a tipos de señales del sistema original
        class_name = class_name.lower()
        
        # Mapeo de clases - ajusta según tu modelo YOLO
        if 'stop' in class_name or 'alto' in class_name:
            return 4  # STOP
        elif 'left' in class_name or 'izquierda' in class_name:
            return 1  # LEFT
        elif 'right' in class_name or 'derecha' in class_name:
            return 2  # RIGHT
        elif 'forward' in class_name or 'straight' in class_name or 'adelante' in class_name:
            return 3  # FORWARD
        elif 'back' in class_name or 'atras' in class_name:
            return 0  # BACK
        elif ('yield' in class_name or 'ceda' in class_name or 'give way' in class_name or 'giveway' in class_name):  # ← CORREGIDO
            return 5  # YIELDv
        elif 'work' in class_name or 'construction' in class_name or 'obras' in class_name:
            return 6  # ROAD_WORK
        else:
            return 3  # Default to FORWARD

# ==================== COMPONENTES ORIGINALES SIN CAMBIOS ====================

class SignType(Enum):
    BACK = 0
    LEFT = 1
    RIGHT = 2
    FORWARD = 3
    STOP = 4
    YIELD = 5
    ROAD_WORK = 6

@dataclass
class Sign:
    type: SignType
    box: np.ndarray
    confidence: float
    approx_dist: Optional[float] = None
    timestamp: Optional[float] = None

class BackgroundPoller:
    #Mock implementation - replace with actual BackgroundPoller if available
    def poll_with_annotated(self, frame, drawing_frame, func):
        return func(drawing_frame)

def group_dotted_lines_simple(points, min_inliers=4, dist_threshold=3.0, distance_ratio=2.5):
    #Groups points into dotted lines
    pts_arr = np.array(points, dtype=float)
    candidate_sets = {}
    for i, j in combinations(range(len(pts_arr)), 2):
        p1, p2 = pts_arr[i], pts_arr[j]
        v = p2 - p1
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            continue
        u = v / norm
        n = np.array([-u[1], u[0]])
        dists = np.abs((pts_arr - p1) @ n)
        inliers = np.where(dists <= dist_threshold)[0]
        if len(inliers) >= min_inliers:
            key = tuple(sorted((points[k] for k in inliers)))
            candidate_sets[key] = inliers

    lines = []
    for key, idxs in candidate_sets.items():
        arr = np.array(key, dtype=float)
        cov = np.cov(arr, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        dir_vec = eigvecs[:, np.argmax(eigvals)]
        proj = arr @ dir_vec
        order = np.argsort(proj)
        sorted_pts = arr[order]
        deltas = np.linalg.norm(np.diff(sorted_pts, axis=0), axis=1)
        if len(deltas) == 0:
            continue
        d_min = deltas.min()
        segments = []
        current = [sorted_pts[0]]
        for pt, gap in zip(sorted_pts[1:], deltas):
            if gap > distance_ratio * d_min:
                if len(current) >= min_inliers:
                    segments.append(current)
                current = [pt]
            else:
                current.append(pt)
        if len(current) >= min_inliers:
            segments.append(current)
        for seg in segments:
            seg_pts = [(int(x), int(y)) for x, y in seg]
            lines.append(seg_pts)
    return lines

class SignDetector:
    def __init__(self, get_signs_func=None, confidence_threshold=0.5, ref_height=50):
        self.confidence_threshold = confidence_threshold
        self._get_signs_func = get_signs_func or (lambda frame, drawing_frame=None: ([], [], [], []))
        self.ref_height = ref_height
        self.chain_length = 8
        self.max_chain_gap = 2
        history_length = max(4, self.chain_length + self.max_chain_gap)
        self._sign_histories = {sign_type: deque(maxlen=history_length) for sign_type in SignType}
        self._bg_poll = BackgroundPoller()

    def get_confirmed_signs_nb(self, frame, drawing_frame=None):
        return self._bg_poll.poll_with_annotated(frame, drawing_frame, lambda af: self.get_confirmed_signs(frame, af))

    def get_confirmed_signs(self, frame, drawing_frame=None) -> list:
        signs = self.set_sign_distances(frame)
        detected_sign_types = {sign.type for sign in signs}
        for sign_type in self._sign_histories:
            self._sign_histories[sign_type].append(sign_type in detected_sign_types)

        def is_confirmed(history):
            return sum(history) >= (self.chain_length - self.max_chain_gap)

        confirmed_signs = []
        for sign in signs:
            if is_confirmed(self._sign_histories[sign.type]):
                confirmed_signs.append(sign)

        if drawing_frame is not None:
            self.draw_signs(drawing_frame, confirmed_signs, color=(0, 255, 0))
        return confirmed_signs

    def set_sign_distances(self, frame, drawing_frame=None) -> list[Sign]:
        signs = self.filter_signs(frame)
        for sign in signs:
            if sign.box is not None and len(sign.box) == 4:
                box_height = abs(sign.box[1] - sign.box[3])
                if box_height > 0:
                    sign.approx_dist = self.ref_height / box_height
        return signs

    def filter_signs(self, frame, drawing_frame=None) -> list[Sign]:
        signs = self.get_signs(frame)
        return [sign for sign in signs if sign.confidence >= self.confidence_threshold]

    def get_signs(self, frame, drawing_frame=None) -> list[Sign]:
        result = self._get_signs_func(frame)
        boxes, sign_types, confidences, class_names = result if result is not None else ([], [], [], [])
        signs = []
        now = time.time()
        for box, sign_type, confidence, class_name in zip(boxes, sign_types, confidences, class_names):
            if int(sign_type) in [item.value for item in SignType]:
                signs.append(Sign(type=SignType(int(sign_type)), box=box, confidence=float(confidence), timestamp=now))
        return signs
    
    def draw_signs(self, frame, signs: list[Sign], color=(0, 255, 0)):
        for sign in signs:
            box = sign.box
            if box is not None and len(box) == 4:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{sign.type.name} ({sign.confidence:.2f})"
                if sign.approx_dist is not None:
                    label += f" [{sign.approx_dist:.2f}m]"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

class IntersectionDetectorAdvanced:
    def __init__(self, v_fov=0.8, min_points=5, max_yaw=30.0, max_thr=0.15):
        self.v_fov = v_fov
        self.morph_kernel = np.ones((3, 3), np.uint8)
        self.erode_iterations = 3
        self.dilate_iterations = 2
        self.max_aspect_ratio = 10.0
        self.min_area = 20
        self.ep = 0.035
        self.min_points = min_points
        self.yaw_threshold = 5.0
        
        # Use SimplePID from original code
        self.w_pid = SimplePID(Kp=2.0, Ki=0.0, Kd=0.1, setpoint=0.0, output_limits=(-math.radians(max_yaw), math.radians(max_yaw)))
        self.v_pid = SimplePID(Kp=0.5, Ki=0.0, Kd=0.1, setpoint=0.7, output_limits=(-max_thr, max_thr))

    def get_dark_mask(self, frame, drawing_frame=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 141, 6)
        
        mask[:int(frame.shape[:2][0] * (1-self.v_fov)), :] = 0
        mask = cv2.erode(mask, kernel=self.morph_kernel, iterations=self.erode_iterations)
        mask = cv2.dilate(mask, kernel=self.morph_kernel, iterations=self.dilate_iterations)
        
        if drawing_frame is not None:
            drawing_frame[:] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return mask

    def find_dots(self, frame, drawing_frame=None):
        mask = self.get_dark_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > self.min_area]
        dots = []
        
        for cnt in contours:
            epsilon = self.ep * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                x, y, w, h = cv2.boundingRect(approx)
                if min(w, h) == 0 or max(w, h) / min(w, h) > self.max_aspect_ratio:
                    continue
                center = (x + w // 2, y + h // 2)
                dots.append(center)
                if drawing_frame is not None:
                    cv2.circle(drawing_frame, center, 5, (0, 0, 255), -1)
                    cv2.polylines(drawing_frame, [approx], True, (0, 255, 0), 2)
        return dots

    def find_dotted_lines(self, frame, drawing_frame=None):
        dots = self.find_dots(frame, drawing_frame=drawing_frame)
        groups = group_dotted_lines_simple(dots, min_inliers=self.min_points)
        dotted_lines = [(group[0], group[-1]) for group in groups if len(group) >= 2]
        line_centers = [((line[0][0] + line[1][0]) // 2, (line[0][1] + line[1][1]) // 2) for line in dotted_lines]
        angles = [((math.degrees(math.atan2(l[1][1] - l[0][1], l[1][0] - l[0][0])) + 90) % 180) - 90 for l in dotted_lines]

        if drawing_frame is not None:
            for i, line in enumerate(dotted_lines):
                cv2.line(drawing_frame, line[0], line[1], (255, 0, 0), 2)
                if i < len(line_centers):
                    cv2.circle(drawing_frame, line_centers[i], 8, (0, 255, 0), -1)
        return dotted_lines, line_centers, angles
    
    def find_intersection(self, frame, drawing_frame=None):
        dotted_lines, centers, angles = self.find_dotted_lines(frame, drawing_frame=drawing_frame)
        if not dotted_lines:
            return None
            
        dotted_lines_data = list(zip(dotted_lines, centers, angles))

        def line_length(line_data):
            (pt1, pt2), center, angle = line_data
            return math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
        
        dotted_lines_data = sorted(dotted_lines_data, key=line_length, reverse=True)
        best_line = dotted_lines_data[0] if dotted_lines_data else None

        if drawing_frame is not None and best_line is not None:
            line, center, angle = best_line
            cv2.drawMarker(drawing_frame, center, (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

        return best_line

    def stop_at_intersection(self, frame, drawing_frame=None, intersection=None):
        throttle, yaw = 0, 0
        intersection = self.find_intersection(frame, drawing_frame=drawing_frame) if intersection is None else intersection

        if intersection is not None:
            line, center, angle = intersection
            error = math.radians(angle)
            yaw = self.w_pid(error)
            alpha = 1 - (abs(error) / self.yaw_threshold) if abs(error) < self.yaw_threshold else 0
            norm_y = center[1] / frame.shape[0]
            measured_distance = (1 - alpha) * self.v_pid.setpoint + alpha * norm_y
            throttle = self.v_pid(measured_distance)

        return throttle, yaw
    
"""class StoplightDetector:
    def __init__(self,
        low_threshold=50,
        high_threshold=150,
        v_fov=0.5,
        min_contour_points=5,
        min_area_ratio=0.9,
        max_area_ratio=1.1,
        min_major_axis=10,
        max_major_axis=200,
        draw_color=(0, 255, 0),
        color_std_thres=45,
        hsv_hue_range=(0, 179),
        hsv_sat_range=(0, 255),
        hsv_val_range=(180, 255),
        yellow_hue=30,
        low_sat_hue_shift=270,
        sat_threshold=50,
        chain_length=4,
        max_chain_gap=1,
        history_len=5,
        solidity_thres=0.92,
        max_eccentricity=0.88,
        brightness_thresh=120,
        min_blob_area=15,
        show_unconfirmed=False,
    ):
        self.show_unconfirmed = show_unconfirmed

        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.v_fov = v_fov
        self.min_contour_points = min_contour_points
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.min_major_axis = min_major_axis
        self.max_major_axis = max_major_axis
        self.draw_color = draw_color
        self.color_std_thres = color_std_thres

        self.hsv_hue_range = hsv_hue_range
        self.hsv_sat_range = hsv_sat_range
        self.hsv_val_range = hsv_val_range
        self.yellow_hue = yellow_hue
        self.low_sat_hue_shift = low_sat_hue_shift
        self.sat_threshold = sat_threshold

        self.chain_length = chain_length
        self.max_chain_gap = max_chain_gap
        color_history_len = max(history_len, chain_length + max_chain_gap)
        self.red_history = deque(maxlen=color_history_len)
        self.yellow_history = deque(maxlen=color_history_len)
        self.green_history = deque(maxlen=color_history_len)

        self.solidity_thres = solidity_thres
        self.max_eccentricity = max_eccentricity
        self.brightness_thresh = brightness_thresh
        self.min_blob_area = min_blob_area

    def classify_stoplight_ellipses(self, frame, drawing_frame=None):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h = int(frame.shape[0] * self.v_fov)
        frame_proc = frame[:h, :]

        gray = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.brightness_thresh, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_ellipses, yellow_ellipses, green_ellipses = [], [], []

        for cnt in contours:
            if cv2.contourArea(cnt) < self.min_blob_area:
                continue
            try:
                ellipse = cv2.fitEllipse(cnt)
            except cv2.error:
                continue
            (center, axes, angle) = ellipse
            major_axis, minor_axis = max(axes), min(axes)
            if not (self.min_major_axis <= major_axis <= self.max_major_axis):
                continue

            # Solidity and eccentricity
            hull = cv2.convexHull(cnt)
            solidity = cv2.contourArea(cnt) / cv2.contourArea(hull)
            if solidity < self.solidity_thres:
                continue
            ecc = np.sqrt(1 - (minor_axis / major_axis) ** 2)
            if ecc > self.max_eccentricity:
                continue

            # Color classification
            mask_ellipse = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.ellipse(mask_ellipse, ellipse, 255, -1)
            pixels = hsv[mask_ellipse == 255]
            if pixels.size == 0:
                continue
            avg_h, avg_s, avg_v = np.mean(pixels, axis=0)
            if avg_s < self.sat_threshold:
                alpha = 1 - (avg_s / self.sat_threshold)
                avg_h = (1 - alpha) * avg_h + alpha * self.yellow_hue

            def hue_dist(h1, h2):
                d = abs(h1 - h2)
                return min(d, 180 - d)

            distances = {
                'red': hue_dist(avg_h, 0),
                'yellow': hue_dist(avg_h, self.yellow_hue),
                'green': hue_dist(avg_h, 65)
            }
            closest = min(distances, key=distances.get)

            if drawing_frame is not None:
                color = {'red': (0, 0, 255), 'yellow': (0, 255, 255), 'green': (0, 255, 0)}[closest]
                cv2.ellipse(drawing_frame, ellipse, color, 2)

            if closest == 'red':
                red_ellipses.append(ellipse)
            elif closest == 'yellow':
                yellow_ellipses.append(ellipse)
            elif closest == 'green':
                green_ellipses.append(ellipse)

        return red_ellipses, yellow_ellipses, green_ellipses

    def identify_stoplight(self, frame, drawing_frame=None):
        red_ellipses, yellow_ellipses, green_ellipses = self.classify_stoplight_ellipses(frame, drawing_frame)

        self.red_history.append(bool(red_ellipses))
        self.yellow_history.append(bool(yellow_ellipses))
        self.green_history.append(bool(green_ellipses))

        def is_confirmed(history):
            return sum(history) >= (self.chain_length - self.max_chain_gap)

        if is_confirmed(self.red_history):
            return 0  # Red
        elif is_confirmed(self.yellow_history):
            return 1  # Yellow
        elif is_confirmed(self.green_history):
            return 2  # Green
        return None"""

class StoplightDetector:
    def __init__(self,
        v_fov=0.9,
        chain_length=4,
        max_chain_gap=1,
        history_len=5,
        show_unconfirmed=False,
    ):
        self.show_unconfirmed = show_unconfirmed
        self.v_fov = v_fov

        self.chain_length = chain_length
        self.max_chain_gap = max_chain_gap
        color_history_len = max(history_len, chain_length + max_chain_gap)
        self.red_history = deque(maxlen=color_history_len)
        self.yellow_history = deque(maxlen=color_history_len)
        self.green_history = deque(maxlen=color_history_len)
        self.frame_count = 0

    def enhance_saturation(self, frame, saturation_factor=1.4):
        """Aumentar saturación para colores más vivos del semáforo"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Aumentar saturación
        s = s.astype(np.float32)
        s = s * saturation_factor
        s = np.clip(s, 0, 255).astype(np.uint8)
        
        # Recombinar
        enhanced_hsv = cv2.merge([h, s, v])
        enhanced_frame = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
        return enhanced_frame

    def filter_background_noise(self, mask, frame_shape):
        """
        Filtrar ruido de fondo como cartón, eliminando áreas grandes y dispersas
        """
        # Encontrar contornos en la máscara
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return mask
        
        # Crear máscara limpia
        clean_mask = np.zeros_like(mask)
        
        # Calcular área total de la imagen para referencia
        total_area = frame_shape[0] * frame_shape[1]
        max_allowed_area = total_area * 0.15  # Máximo 15% de la imagen
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filtrar por área (no muy grande ni muy pequeña)
            if 50 < area < max_allowed_area:
                # Calcular solidity manualmente (área del contorno / área del hull convexo)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                
                if hull_area > 0:
                    solidity = area / hull_area
                    
                    # Filtrar por solidity (qué tan "sólido" es)
                    if solidity > 0.3:  # Evita formas muy dispersas
                        # Verificar aspect ratio del bounding box
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 10
                        
                        # Solo mantener formas no muy alargadas
                        if aspect_ratio < 5.0:
                            cv2.fillPoly(clean_mask, [contour], 255)
        
        return clean_mask

    def is_likely_traffic_light(self, contour, frame, color_type="yellow"):
        """
        Determinar si un contorno es realmente un semáforo y no fondo de cartón
        """
        area = cv2.contourArea(contour)
        
        # 1. Filtro básico de área
        if area < 100 or area > 8000:  # Muy pequeño o muy grande
            return False, "área fuera de rango"
        
        # 2. Crear máscara para analizar solo esta región
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        
        # 3. Extraer región y convertir a diferentes espacios de color
        region_bgr = cv2.bitwise_and(frame, frame, mask=mask)
        region_hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
        region_lab = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2LAB)
        
        # 4. Obtener solo los píxeles del contorno
        pixels_hsv = region_hsv[mask > 0]
        pixels_lab = region_lab[mask > 0]
        
        if len(pixels_hsv) == 0:
            return False, "sin píxeles"
        
        # 5. Estadísticas de color
        mean_h = np.mean(pixels_hsv[:, 0])
        mean_s = np.mean(pixels_hsv[:, 1])
        mean_v = np.mean(pixels_hsv[:, 2])
        std_v = np.std(pixels_hsv[:, 2])
        std_s = np.std(pixels_hsv[:, 1])
        
        # 6. Análisis en espacio LAB (mejor para distinguir amarillo vs beige)
        mean_a = np.mean(pixels_lab[:, 1])  # Canal a* (verde-rojo)
        mean_b = np.mean(pixels_lab[:, 2])  # Canal b* (azul-amarillo)
        
        if color_type == "yellow":
            # 7. Criterios específicos para amarillo vs cartón:
            
            # A. El amarillo verdadero tiene mayor saturación
            if mean_s < 80:
                return False, f"saturación muy baja: {mean_s:.0f}"
            
            # B. Los LEDs amarillos tienen más variación en brillo
            if std_v < 15:
                return False, f"brillo muy uniforme: {std_v:.1f}"
            
            # C. En espacio LAB, amarillo verdadero tiene b* > 0 (hacia amarillo)
            if mean_b < 10:
                return False, f"no es amarillo en LAB: {mean_b:.1f}"
            
            # D. Verificar que no sea demasiado "hacia el verde" en LAB
            if mean_a < -10:  # Muy hacia el verde
                return False, f"muy verde en LAB: {mean_a:.1f}"
            
            # E. El brillo debe ser alto para LEDs
            if mean_v < 160:
                return False, f"brillo insuficiente: {mean_v:.0f}"
            
            # F. Verificar que tenga forma compacta
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > 2.5:  # Muy alargado
                return False, f"muy alargado: {aspect_ratio:.1f}"
            
            # G. Verificar que no esté en el borde de la imagen (cartón suele estar ahí)
            img_h, img_w = frame.shape[:2]
            center_x, center_y = x + w//2, y + h//2
            
            border_margin = 50
            if (center_x < border_margin or center_x > img_w - border_margin or
                center_y < border_margin or center_y > img_h - border_margin):
                return False, "muy cerca del borde"
        
        return True, f"OK (S:{mean_s:.0f}, V:{mean_v:.0f}, stdV:{std_v:.1f}, LAB_b:{mean_b:.1f})"

    def analyze_shape(self, contour, min_area=100, color_type="general", original_frame=None):
        """
        Analizar si un contorno es una forma válida (círculo o elipse)
        """
        area = cv2.contourArea(contour)
        
        if area < min_area:
            return False, f"Área muy pequeña: {area:.0f}"
        
        # Análisis específico para distinguir semáforo de fondo
        if original_frame is not None:
            is_light, light_info = self.is_likely_traffic_light(contour, original_frame, color_type)
            if not is_light:
                return False, f"No es semáforo: {light_info}"
        
        # Obtener el perímetro
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False, "Perímetro cero"
        
        # Calcular circularidad (4π * área / perímetro²)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Ajustar elipse si hay suficientes puntos
        if len(contour) >= 5:
            try:
                # Ajustar elipse al contorno
                ellipse = cv2.fitEllipse(contour)
                center, axes, angle = ellipse
                major_axis = max(axes)
                minor_axis = min(axes)
                
                # Calcular excentricidad de la elipse
                if major_axis > 0:
                    eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
                    aspect_ratio = major_axis / minor_axis
                else:
                    eccentricity = 1.0
                    aspect_ratio = 1.0
                
                # Criterios de validación más estrictos para amarillo
                if color_type == "yellow":
                    # Amarillo MUY restrictivo para evitar cartón
                    is_valid_ellipse = (
                        eccentricity <= 0.6 and   # Solo elipses moderadas
                        aspect_ratio <= 2.0 and   # Evita formas alargadas
                        circularity > 0.4         # Alta circularidad requerida
                    )
                else:
                    # Criterios normales para rojo y verde
                    is_valid_ellipse = (
                        eccentricity <= 0.8 and
                        aspect_ratio <= 3.0 and
                        circularity > 0.2
                    )
                
                shape_info = f"área:{area:.0f}, circ:{circularity:.2f}, exc:{eccentricity:.2f}, ratio:{aspect_ratio:.2f} | {light_info if original_frame else 'sin análisis'}"
                
                return is_valid_ellipse, shape_info
                
            except cv2.error:
                # Si falla el ajuste de elipse, usar solo circularidad
                min_circularity = 0.5 if color_type == "yellow" else 0.3
                is_valid_circle = circularity > min_circularity
                shape_info = f"área:{area:.0f}, circ:{circularity:.2f} (solo círculo) | {light_info if original_frame else 'sin análisis'}"
                return is_valid_circle, shape_info
        else:
            # Pocos puntos, usar solo circularidad
            min_circularity = 0.5 if color_type == "yellow" else 0.3
            is_valid_circle = circularity > min_circularity
            shape_info = f"área:{area:.0f}, circ:{circularity:.2f} (pocos puntos) | {light_info if original_frame else 'sin análisis'}"
            return is_valid_circle, shape_info

    def draw_shape_info(self, drawing_frame, contour, color, label, shape_info, y_pos):
        """Dibujar información de la forma detectada"""
        if drawing_frame is not None:
            # Dibujar contorno
            cv2.drawContours(drawing_frame, [contour], -1, color, 3)
            
            # Dibujar elipse ajustada si es posible
            if len(contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(contour)
                    cv2.ellipse(drawing_frame, ellipse, color, 2)
                except cv2.error:
                    pass
            
            # Texto con información
            cv2.putText(drawing_frame, f"{label}: {shape_info}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def classify_stoplight_colors(self, frame, drawing_frame=None):
        """Detectar colores de semáforo usando la lógica que funciona"""
        # Mejorar saturación para mejor detección
        enhanced_frame = self.enhance_saturation(frame, saturation_factor=1.4)
        hsv = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2HSV)
        
        # Aplicar ROI vertical
        h = int(frame.shape[0] * self.v_fov)
        frame_proc = enhanced_frame[:h, :]
        hsv_proc = hsv[:h, :]

        # Rangos HSV exactos del código que funciona
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])

        # ROJO - rangos más restrictivos para eliminar falsos positivos
        lower_red1 = np.array([0, 120, 100])
        upper_red1 = np.array([8, 255, 255])
        lower_red2 = np.array([172, 120, 100])
        upper_red2 = np.array([180, 255, 255])

        # Amarillo - rangos MÁS específicos para evitar cartón
        # Solo amarillo verdadero con cierta saturación
        lower_yellow1 = np.array([20, 100, 150])   # Saturación mínima alta
        upper_yellow1 = np.array([30, 255, 255])   # Rango estrecho
        
        # Amarillo brillante (LEDs)
        lower_yellow2 = np.array([18, 80, 200])    # Alto brillo, saturación media
        upper_yellow2 = np.array([32, 200, 255])   # Evita amarillos muy pálidos

        # Crear máscaras
        mask_green = cv2.inRange(hsv_proc, lower_green, upper_green)
        mask_red1 = cv2.inRange(hsv_proc, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv_proc, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        mask_yellow1 = cv2.inRange(hsv_proc, lower_yellow1, upper_yellow1)
        mask_yellow2 = cv2.inRange(hsv_proc, lower_yellow2, upper_yellow2)
        mask_yellow = cv2.bitwise_or(mask_yellow1, mask_yellow2)
        
        # Filtrar ruido de fondo específicamente para amarillo
        mask_yellow = self.filter_background_noise(mask_yellow, frame_proc.shape)

        # Operaciones morfológicas
        kernel = np.ones((5, 5), np.uint8)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)

        # Buscar contornos
        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        red_detected = False
        yellow_detected = False
        green_detected = False

        # Debug: Número de contornos encontrados
        if len(contours_green) > 0 or len(contours_red) > 0 or len(contours_yellow) > 0:
            print(f'🔍 Contornos encontrados - Verde: {len(contours_green)}, Rojo: {len(contours_red)}, Amarillo: {len(contours_yellow)}')

        colors_detected = []

        # 🟡 AMARILLO - Detección ultra-estricta anti-cartón
        if contours_yellow:
            # Pre-filtrar contornos por área más estricta
            valid_contours = []
            for c in contours_yellow:
                area = cv2.contourArea(c)
                if 150 < area < 3000:  # Rango más estricto
                    # Verificar forma básica antes del análisis completo
                    perimeter = cv2.arcLength(c, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.3:  # Pre-filtro de circularidad
                            valid_contours.append(c)
            
            if valid_contours:
                # Tomar el más circular, no necesariamente el más grande
                best_contour = max(valid_contours, key=lambda c: 
                    4 * np.pi * cv2.contourArea(c) / (cv2.arcLength(c, True) ** 2))
                
                is_valid, shape_info = self.analyze_shape(best_contour, min_area=150, 
                                                        color_type="yellow", original_frame=frame_proc)
                
                if is_valid:
                    yellow_detected = True
                    colors_detected.append(f"AMARILLO ({shape_info})")
                    self.draw_shape_info(drawing_frame, best_contour, (0, 255, 255), 
                                       "YELLOW", shape_info, 80)
                else:
                    # Debug: mostrar por qué se rechazó
                    if self.frame_count % 30 == 0:  # Cada 30 frames
                        print(f"❌ Amarillo rechazado: {shape_info}")

        # 🟥 ROJO - Detección con análisis de forma
        if contours_red:
            largest_red = max(contours_red, key=cv2.contourArea)
            is_valid, shape_info = self.analyze_shape(largest_red, min_area=150)
            
            if is_valid:
                red_detected = True
                colors_detected.append(f"ROJO ({shape_info})")
                self.draw_shape_info(drawing_frame, largest_red, (0, 0, 255), 
                                   "RED", shape_info, 50)

        # 🟢 VERDE - Detección con análisis de forma
        if contours_green:
            largest_green = max(contours_green, key=cv2.contourArea)
            is_valid, shape_info = self.analyze_shape(largest_green, min_area=200)
            
            if is_valid:
                green_detected = True
                colors_detected.append(f"VERDE ({shape_info})")
                self.draw_shape_info(drawing_frame, largest_green, (0, 255, 0), 
                                   "GREEN", shape_info, 110)

        # Imprimir colores detectados
        if colors_detected:
            print(f'🎯 Colores detectados: {", ".join(colors_detected)}')
        else:
            if self.frame_count % 60 == 0:
                print('❌ No se detectaron colores válidos')

        return red_detected, yellow_detected, green_detected

    def identify_stoplight(self, frame, drawing_frame=None):
        """Identificar el estado del semáforo con confirmación por historial"""
        red_detected, yellow_detected, green_detected = self.classify_stoplight_colors(frame, drawing_frame)

        # Actualizar historiales
        self.red_history.append(red_detected)
        self.yellow_history.append(yellow_detected)
        self.green_history.append(green_detected)

        def is_confirmed(history):
            return sum(history) >= (self.chain_length - self.max_chain_gap)

        # Verificar estado confirmado
        confirmed_red = is_confirmed(self.red_history)
        confirmed_yellow = is_confirmed(self.yellow_history)
        confirmed_green = is_confirmed(self.green_history)

        # Mostrar detecciones no confirmadas si está habilitado
        if drawing_frame is not None:
            if self.show_unconfirmed:
                status = f"R:{red_detected} Y:{yellow_detected} G:{green_detected}"
                cv2.putText(drawing_frame, status, (10, 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Mostrar estado confirmado
            if confirmed_red:
                cv2.putText(drawing_frame, "STOPLIGHT: RED", (10, 170), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            elif confirmed_yellow:
                cv2.putText(drawing_frame, "STOPLIGHT: YELLOW", (10, 170), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            elif confirmed_green:
                cv2.putText(drawing_frame, "STOPLIGHT: GREEN", (10, 170), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Incrementar contador de frames
        self.frame_count += 1

        # Retornar estado confirmado
        if confirmed_red:
            return 0  # Red
        elif confirmed_yellow:
            return 1  # Yellow
        elif confirmed_green:
            return 2  # Green
        return None  # No detection

class FlagDetector:
    def __init__(self, dist_thres=0.40):
        self.dist_thres = dist_thres
        self.end_reached = False
        
        # Parámetros de cámara (copiar del nodo separado)
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

    def undistort_frame(self, frame):
        return cv2.undistort(frame, self.K, self.D)

    def get_checkerboard_corners(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
        return corners if ret else None

    def estimate_distance_from_height(self, corners):
        ys = corners[:, :, 1].flatten()
        h_pix = ys.max() - ys.min()
        H_real = self.square_size * (self.pattern_size[1] - 1)
        Z = (self.f_y * H_real) / h_pix
        return Z

    def get_flag_distance_nb(self, frame, drawing_frame=None):
        """Detecta bandera y retorna distancia"""
        undist = self.undistort_frame(frame)
        corners = self.get_checkerboard_corners(undist)
        
        if corners is not None:
            distance = self.estimate_distance_from_height(corners)
            
            # Dibujar en debug frame si está disponible
            if drawing_frame is not None:
                cv2.drawChessboardCorners(drawing_frame, self.pattern_size, corners, True)
                status_text = f"FLAG: {distance:.2f}m"
                color = (0, 255, 0) if distance <= self.dist_thres else (0, 0, 255)
                cv2.putText(drawing_frame, status_text, (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            return distance
        return None

class TrackNavigator:
    def __init__(self, line_follower=None, sign_detector=None, intersection_detector=None,
             stoplight_detector=None, flag_detector=None, end_action=None, ongoing_end_action=None):
        self.lf = line_follower
        self.sd = sign_detector
        self.id = intersection_detector
        self.stl_det = stoplight_detector
        self.fd = flag_detector
        self.current_light = -1  # -1: desconocido, 0: rojo, 1: amarillo, 2: verde

        self.end_action = end_action
        self.ongoing_end_action = ongoing_end_action
        # Static variables
        self.last_signs: list[Sign] = []
        self.last_turn_sign: Sign = None
        self.yielding = False
        self.poll = True
        self.turn_age = 5  # seconds
        self.stopping = False
        self.stop_time = None
        self.stop_duration = 2.0  # segundos detenido
        # Estados de la máquina de estados
        self.state = "FOLLOWING"  # FOLLOWING, INTERSECTION_DETECTED, CROSSING, TURNING, RESUMING
        self.intersection_confidence = 0
        self.min_confidence = 3
        
        # Variables para manejo de intersecciones
        self.turn_command = None  # Dirección a tomar (LEFT, RIGHT, FORWARD, BACK)
        self.crossing_timer = 0
        self.turning_timer = 0
        self.resuming_timer = 0
        
        # Configuración de tiempos (en frames/ciclos)
        self.crossing_duration = 28    # Tiempo para cruzar la intersección (1 segundo a 30fps)
        self.turning_duration = 30     # Tiempo para hacer el giro (1.5 segundos)
        self.resuming_duration = 15    # Tiempo para estabilizarse (0.5 segundos)
        self.pending_action = None
        self.just_crossed_intersection = False
        self.cross_complete_time = None
        self.intersection_detected = False
        self.poll = True
        self.last_stoplight = None
        if self.lf:
            self.lf.authority = 1.0



    def navigate(self, frame, drawing_frame=None):
        """Máquina de estados para navegación completa con intersecciones - VERSIÓN CORREGIDA"""
        
        # --- DETECCIÓN DE BANDERA ---
        if self.fd:
            flag_dist = self.fd.get_flag_distance_nb(frame, drawing_frame=drawing_frame)
            if flag_dist is not None and flag_dist <= self.fd.dist_thres:
                self.fd.end_reached = True
                if self.end_action:
                    self.end_action()
                return 0.0, 0.0

            if self.fd.end_reached:
                if self.ongoing_end_action:
                    ret = self.ongoing_end_action()
                    if isinstance(ret, tuple) and len(ret) == 2:
                        return ret
                return 0.0, 0.0

        # --- DETECCIÓN DE SEMÁFORO (MEJORADA Y UNIFICADA) ---
        self.current_light = None
        self.last_stoplight = None
        
        if self.stl_det:
            # Detectar semáforo UNA SOLA VEZ por frame
            detected_light = self.stl_det.identify_stoplight(frame, drawing_frame=drawing_frame)
            self.current_light = detected_light
            self.last_stoplight = detected_light

        # 🚨 VERIFICACIÓN CRÍTICA DE SEMÁFORO ROJO - MÁXIMA PRIORIDAD
        # Esta verificación debe ser PRIORITARIA sobre cualquier otro estado
        if self.current_light == 0:  # ROJO DETECTADO
            # FORZAR DETENCIÓN INMEDIATA, sin importar el estado actual
            if drawing_frame is not None:
                cv2.putText(drawing_frame, "🚨 RED LIGHT - EMERGENCY STOP", 
                        (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                # Dibujar rectángulo de alerta
                cv2.rectangle(drawing_frame, (0, 240), (500, 280), (0, 0, 255), 3)
            
            # Log crítico
            print("🚨 CRITICAL STOP: RED LIGHT DETECTED!")
            return 0.0, 0.0  # DETENCIÓN ABSOLUTA - NO PROCESAR MÁS ESTADOS

        # Si hay luz amarilla, aplicar lógica conservadora
        if self.current_light == 1:  # AMARILLO
            if self.state == "INTERSECTION_DETECTED":
                # Política conservadora: con amarillo NO cruzar
                if drawing_frame is not None:
                    cv2.putText(drawing_frame, "🟡 YELLOW - CONSERVATIVE STOP", 
                            (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                print("🟡 Yellow light - stopping conservatively")
                return 0.0, 0.0  # Detenerse por seguridad
            else:
                # Si no estamos en intersección, reducir velocidad significativamente
                if hasattr(self.lf, 'authority'):
                    self.lf.authority = 0.3

        # --- MANEJO DE SEÑALES DE STOP ---
        if self.stopping:
            if self.stop_time is None:
                self.stop_time = time.time()
                return 0.0, 0.0
            elif time.time() - self.stop_time < self.stop_duration:
                return 0.0, 0.0
            else:
                self.stopping = False
                self.stop_time = None

        # --- APLICAR ACCIONES PENDIENTES POST-INTERSECCIÓN ---
        if self.just_crossed_intersection and self.cross_complete_time is not None:
            if time.time() > self.cross_complete_time:
                self.just_crossed_intersection = False
                if self.pending_action == 'STOP':
                    self.stopping = True
                elif self.pending_action in ['YIELD', 'SLOW']:
                    if hasattr(self.lf, 'authority'):
                        self.lf.authority = 0.5
                self.pending_action = None

        # --- DETECCIÓN DE SEÑALES ---
        if self.sd:
            self.last_signs = self.sd.get_confirmed_signs_nb(frame, drawing_frame=drawing_frame)
            # Actualizar comando de giro si hay señales direccionales
            if self.last_signs:
                turn_signs = [s for s in self.last_signs if 0 <= s.type.value <= 3]
                if turn_signs:
                    # Tomar la señal con mayor confianza
                    best_turn_sign = max(turn_signs, key=lambda s: s.confidence)
                    self.turn_command = best_turn_sign.type
                    self.last_turn_sign = best_turn_sign

        # --- DETECCIÓN DE INTERSECCIONES ---
        intersection = None
        if self.id:
            intersection = self.id.find_intersection(frame, drawing_frame=drawing_frame)
            
            # Sistema de confianza
            if intersection:
                self.intersection_confidence = min(self.intersection_confidence + 1, self.min_confidence + 2)
            else:
                self.intersection_confidence = max(self.intersection_confidence - 1, 0)

        # ============ MÁQUINA DE ESTADOS MEJORADA ============
        
        if self.state == "FOLLOWING":
            # Estado normal: seguir líneas
            if hasattr(self.lf, 'authority'):
                # Restaurar authority a 1.0 si no hay restricciones de semáforo
                if self.current_light != 1:  # Si no es amarillo
                    self.lf.authority = 1.0
            
            # Controlar velocidad basado en señales
            self._apply_sign_speed_control()
            
            # Verificar si hay intersección detectada
            if self.intersection_confidence >= self.min_confidence and intersection:
                if self.turn_command:  # Solo si tenemos una señal de dirección
                    self.state = "INTERSECTION_DETECTED"
                    if drawing_frame is not None:
                        cv2.putText(drawing_frame, f"INTERSECTION DETECTED - WILL GO {self.turn_command.name}", 
                                (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Seguir línea normalmente
            throttle, yaw = self._follow_line_with_authority(frame, drawing_frame)
            
        elif self.state == "INTERSECTION_DETECTED":
            # Detectamos intersección, prepararse para cruzar
            if hasattr(self.lf, 'authority'):
                self.lf.authority = 0.8  # Reducir velocidad para aproximación
            
            # Alinearse con la intersección
            if intersection:
                throttle, yaw = self.id.stop_at_intersection(frame, drawing_frame=drawing_frame, intersection=intersection)
                
                # 🚨 CONDICIÓN DE CRUCE MEJORADA CON VERIFICACIÓN ESTRICTA DE SEMÁFORO
                if abs(throttle) < 0.05:  # Estamos muy cerca de la intersección
                    
                    if self.current_light == 2:  # SOLO VERDE PERMITE CRUZAR
                        self.state = "CROSSING"
                        self.crossing_timer = self.crossing_duration
                        if drawing_frame is not None:
                            cv2.putText(drawing_frame, f"🟢 GREEN - CROSSING {self.turn_command.name}", 
                                    (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        print(f"🟢 Green light confirmed - starting to cross towards {self.turn_command.name}")
                    
                    elif self.current_light == 0:  # ROJO - MANTENER PARADO
                        throttle = 0.0
                        yaw = 0.0
                        if drawing_frame is not None:
                            cv2.putText(drawing_frame, "🔴 RED - MUST STOP", 
                                    (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
                        print("🔴 Red light - maintaining stop at intersection")
                    
                    elif self.current_light == 1:  # AMARILLO - NO CRUZAR
                        throttle = 0.0
                        yaw = 0.0
                        if drawing_frame is not None:
                            cv2.putText(drawing_frame, "🟡 YELLOW - SAFETY STOP", 
                                    (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        print("🟡 Yellow light - safety stop at intersection")
                    
                    elif self.current_light is None:  # SIN SEMÁFORO DETECTADO
                        # Ser muy conservador: esperar confirmación de que NO hay semáforo
                        if not hasattr(self, '_no_light_counter'):
                            self._no_light_counter = 0
                        self._no_light_counter += 1
                        
                        # Requiere muchos frames consecutivos sin detectar semáforo
                        if self._no_light_counter > 15:  # 15 frames sin detectar semáforo
                            self.state = "CROSSING"
                            self.crossing_timer = self.crossing_duration
                            if drawing_frame is not None:
                                cv2.putText(drawing_frame, f"NO TRAFFIC LIGHT - CROSSING {self.turn_command.name}", 
                                        (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            print(f"No traffic light confirmed after {self._no_light_counter} frames - proceeding to cross")
                        else:
                            throttle = 0.0
                            yaw = 0.0
                            if drawing_frame is not None:
                                cv2.putText(drawing_frame, f"WAITING FOR LIGHT CONFIRMATION {self._no_light_counter}/15", 
                                        (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            print(f"Waiting for traffic light confirmation: {self._no_light_counter}/15")
                else:
                    # Reset contador si no estamos cerca de la intersección
                    if hasattr(self, '_no_light_counter'):
                        self._no_light_counter = 0
            else:
                # Si perdemos la intersección, volver a seguir línea
                throttle, yaw = self._follow_line_with_authority(frame, drawing_frame)
                
        elif self.state == "CROSSING":
            # Cruzar la intersección en línea recta
            # Una vez iniciado el cruce, DEBE completarse sin revisar semáforo
            # (es peligroso detenerse en medio de la intersección)
            throttle = 0.15  # Velocidad constante para cruzar
            yaw = 0.0       # Sin giro, ir derecho
            
            self.crossing_timer -= 1
            if self.crossing_timer <= 0:
                self.state = "TURNING"
                self.turning_timer = self.turning_duration
                if drawing_frame is not None:
                    cv2.putText(drawing_frame, f"CROSSED - NOW TURNING {self.turn_command.name}", 
                            (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                print(f"Intersection crossed - now turning {self.turn_command.name}")
            
            if drawing_frame is not None:
                cv2.putText(drawing_frame, f"CROSSING INTERSECTION: {self.crossing_timer}", 
                        (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
        elif self.state == "TURNING":
            # Ejecutar el giro según la señal
            throttle, yaw = self._execute_turn(frame, drawing_frame)
            
            self.turning_timer -= 1

            if self.turning_timer <= 0:
                self.state = "RESUMING"
                self.resuming_timer = self.resuming_duration
                self.just_crossed_intersection = True
                self.cross_complete_time = time.time() + 1.5  # 1.5s después del giro

                if drawing_frame is not None:
                    cv2.putText(drawing_frame, "TURN COMPLETED - RESUMING LINE FOLLOWING", 
                            (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                print("Turn completed - resuming line following")
            
            if drawing_frame is not None:
                cv2.putText(drawing_frame, f"TURNING {self.turn_command.name}: {self.turning_timer}", 
                        (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
        elif self.state == "RESUMING":
            # Estabilizar y volver a seguimiento normal
            throttle, yaw = self._follow_line_with_authority(frame, drawing_frame)
            
            self.resuming_timer -= 1
            if self.resuming_timer <= 0:
                self.state = "FOLLOWING"
                self.turn_command = None  # Limpiar comando
                self.intersection_confidence = 0  # Reset confianza
                if hasattr(self, '_no_light_counter'):
                    self._no_light_counter = 0  # Reset contador de confirmación
                if drawing_frame is not None:
                    cv2.putText(drawing_frame, "RESUMED NORMAL LINE FOLLOWING", 
                            (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                print("Resumed normal line following")
            
            if drawing_frame is not None:
                cv2.putText(drawing_frame, f"RESUMING: {self.resuming_timer}", 
                        (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # ============ INFORMACIÓN DE DEBUG MEJORADA ============
        if drawing_frame is not None:
            # Estado actual con color según criticidad
            state_colors = {
                "FOLLOWING": (0, 255, 0),
                "INTERSECTION_DETECTED": (0, 255, 255),
                "CROSSING": (255, 0, 255),
                "TURNING": (255, 255, 0),
                "RESUMING": (0, 255, 0)
            }
            state_color = state_colors.get(self.state, (255, 255, 255))
            
            cv2.putText(drawing_frame, f"STATE: {self.state}", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
            
            # Comando de giro actual
            if self.turn_command:
                cv2.putText(drawing_frame, f"TURN CMD: {self.turn_command.name}", (10, 180), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Confianza de intersección
            cv2.putText(drawing_frame, f"INT_CONF: {self.intersection_confidence}/{self.min_confidence}", 
                    (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 🚨 MOSTRAR ESTADO DEL SEMÁFORO DE FORMA MUY PROMINENTE
            if self.current_light is not None:
                light_names = ['🔴 RED', '🟡 YELLOW', '🟢 GREEN']
                light_colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0)]
                light_bg_colors = [(0, 0, 128), (0, 128, 128), (0, 128, 0)]
                
                if 0 <= self.current_light < len(light_names):
                    # Fondo del texto para mayor visibilidad
                    text_size = cv2.getTextSize(f"STOPLIGHT: {light_names[self.current_light]}", 
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
                    cv2.rectangle(drawing_frame, (5, 280), (text_size[0] + 15, 320), 
                                light_bg_colors[self.current_light], -1)
                    
                    cv2.putText(drawing_frame, f"STOPLIGHT: {light_names[self.current_light]}", 
                            (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 1.0, light_colors[self.current_light], 3)
            else:
                cv2.rectangle(drawing_frame, (5, 280), (250, 320), (64, 64, 64), -1)
                cv2.putText(drawing_frame, "STOPLIGHT: NONE DETECTED", 
                        (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
            
            # Mostrar velocidades actuales
            cv2.putText(drawing_frame, f"v: {throttle:.3f} m/s", (10, 350), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(drawing_frame, f"w: {math.degrees(yaw):.1f}°/s", (10, 370), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Authority del line follower
            if hasattr(self.lf, 'authority'):
                cv2.putText(drawing_frame, f"Authority: {self.lf.authority:.2f}", (10, 390), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # ============ VALIDACIÓN FINAL DE SEGURIDAD ============
        # Verificación final por si algo se filtró
        if self.current_light == 0 and throttle > 0:
            print("🚨 FINAL SAFETY CHECK: Overriding movement due to red light!")
            throttle = 0.0
            yaw = 0.0
        
        return throttle, yaw

    """def _apply_sign_speed_control(self):
        #Aplicar control de velocidad basado en señales detectadas
        if hasattr(self.lf, 'authority'):
            self.lf.authority = 1.0  # Reset
        
        if self.last_signs:
            closest_signs = {}
            for sign in self.last_signs:
                if sign.approx_dist and (sign.type not in closest_signs or 
                                       sign.approx_dist < closest_signs[sign.type].approx_dist):
                    closest_signs[sign.type] = sign

            if SignType.STOP in closest_signs and closest_signs[SignType.STOP].approx_dist < 0.4:
                self.stopping = True
            elif SignType.ROAD_WORK in closest_signs and closest_signs[SignType.ROAD_WORK].approx_dist < 0.75:
                if hasattr(self.lf, 'authority'):
                    self.lf.authority = 0.5  # Reducir velocidad
            elif SignType.YIELD in closest_signs and closest_signs[SignType.YIELD].approx_dist < 0.75:
                if hasattr(self.lf, 'authority'):
                    self.lf.authority = 0.5  # Reducir velocidad a la mitad"""
    """def _apply_sign_speed_control(self):
        # Guardar señales detectadas en pending_action si estamos en FOLLOWING
        if self.state == "FOLLOWING" and self.last_signs:
            closest_signs = {}
            for sign in self.last_signs:
                if sign.approx_dist and (sign.type not in closest_signs or 
                                        sign.approx_dist < closest_signs[sign.type].approx_dist):
                    closest_signs[sign.type] = sign

            if SignType.STOP in closest_signs and closest_signs[SignType.STOP].approx_dist < 0.7:
                self.pending_action = 'STOP'
            elif SignType.ROAD_WORK in closest_signs and closest_signs[SignType.ROAD_WORK].approx_dist < 0.75:
                self.pending_action = 'SLOW'
            elif SignType.YIELD in closest_signs and closest_signs[SignType.YIELD].approx_dist < 0.75:
                self.pending_action = 'YIELD'"""
    def _apply_sign_speed_control(self):
        if self.last_signs:
            closest_signs = {}
            for sign in self.last_signs:
                if sign.approx_dist and (sign.type not in closest_signs or 
                                        sign.approx_dist < closest_signs[sign.type].approx_dist):
                    closest_signs[sign.type] = sign

            # Si viene una intersección pronto (en FOLLOWING), guardar la acción
            if self.state == "FOLLOWING" and self.intersection_detected:
                if SignType.STOP in closest_signs and closest_signs[SignType.STOP].approx_dist < 0.7:
                    self.pending_action = 'STOP'
                elif SignType.ROAD_WORK in closest_signs and closest_signs[SignType.ROAD_WORK].approx_dist < 0.75:
                    self.pending_action = 'SLOW'
                elif SignType.YIELD in closest_signs and closest_signs[SignType.YIELD].approx_dist < 0.75:
                    self.pending_action = 'YIELD'
            
            # Si no hay intersección, aplicar directamente
            else:
                if SignType.STOP in closest_signs and closest_signs[SignType.STOP].approx_dist < 0.7:
                    self.stopping = True
                elif SignType.ROAD_WORK in closest_signs and closest_signs[SignType.ROAD_WORK].approx_dist < 0.75:
                    if hasattr(self.lf, 'authority'):
                        self.lf.authority = 0.5
                elif SignType.YIELD in closest_signs and closest_signs[SignType.YIELD].approx_dist < 0.75:
                    if hasattr(self.lf, 'authority'):
                        self.lf.authority = 0.5


    def _follow_line_with_authority(self, frame, drawing_frame):
        #Seguir línea con el sistema de authority
        if self.lf and hasattr(self.lf, 'follow_line'):
            return self.lf.follow_line(frame, drawing_frame=drawing_frame)
        else:
            return 0.0, 0.0

    def _execute_turn(self, frame, drawing_frame):
        #Ejecutar giro según el comando de dirección
        throttle = 0.1  # Velocidad lenta durante el giro
        
        if self.turn_command == SignType.LEFT:
            yaw = math.radians(45)  # Girar a la izquierda
        elif self.turn_command == SignType.RIGHT:
            yaw = math.radians(-45)  # Girar a la derecha
        elif self.turn_command == SignType.BACK:
            yaw = math.radians(90)   # Giro de 180 grados (más tiempo)
            if self.turning_timer > self.turning_duration * 0.7:  # Primera parte del giro
                yaw = math.radians(60)
        elif self.turn_command == SignType.FORWARD:
            yaw = 0.0  # Continuar recto
            throttle = 0.15  # Puede ir un poco más rápido
        else:
            yaw = 0.0
            
        return throttle, yaw

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
        #Reinicia el controlador PID
        self._integral = 0.0
        self._last_error = 0.0
        self._last_time = None

# ==================== FUNCIONES PARA DETECCIÓN DE INTERSECCIONES ORIGINALES ====================

def find_dots_for_intersection(frame, drawing_frame=None):
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

# ==================== NODO PRINCIPAL CON YOLO INTEGRADO ====================

class LineFollowerWithYOLONode(Node):
    def __init__(self):
        super().__init__('line_follower_with_yolo_node')

        # Parámetros YOLO
        self.declare_parameter('model_path', 'traffic_signs_dataset/best.pt')
        self.declare_parameter('confidence_threshold', 0.6)
        self.declare_parameter('use_yolo', True)


        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.use_yolo = self.get_parameter('use_yolo').get_parameter_value().bool_value

        self.bridge = CvBridge()
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_safe', 10)
        #self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10) Simulador
        self.image_sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        #self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10) Simulador
        self.debug_pub = self.create_publisher(Image, '/debug_image', 10)
        self.status_pub = self.create_publisher(String, '/robot_status', 10)

        # Inicializar YOLO si está habilitado
        self.yolo_detector = None
        if self.use_yolo:
            self.yolo_detector = YOLOSignDetector(model_path, confidence_threshold)
            if self.yolo_detector.model is not None:
                self.get_logger().info('✅ YOLO detector initialized successfully')
            else:
                self.get_logger().error('❌ Failed to initialize YOLO detector')
                self.use_yolo = False

        # Parámetros del seguidor de líneas original (sin cambios)
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

        # PIDs para control de intersección original
        self.intersection_yaw_pid = SimplePID(Kp=1.5, Ki=0.0, Kd=0.1, setpoint=0.0, 
                                            output_limits=(-math.radians(30), math.radians(30)))
        self.intersection_speed_pid = SimplePID(Kp=0.3, Ki=0.0, Kd=0.05, setpoint=0.7, 
                                              output_limits=(-0.15, 0.15))
        
        # Estados de navegación original
        self.intersection_detected = False
        self.intersection_confidence = 0
        self.min_confidence = 3
        self.last_intersection = None
        self.intersection_timeout = 0
        self.stoplight_detector = StoplightDetector()
        self.flag_detector = FlagDetector(dist_thres=0.40)
        # Modo de operación
        self.mode = "FOLLOWING"  # "FOLLOWING", "INTERSECTION", "ENHANCED"

        # ============ COMPONENTES CON YOLO ============
        # Función de detección YOLO para integrar con el sistema original
        def get_signs_func(frame, drawing_frame=None):
            if self.yolo_detector:
                return self.yolo_detector.detect_signs(frame)
            return [], [], [], []

        # Inicializar detectores con YOLO
        self.sign_detector = SignDetector(
            get_signs_func=get_signs_func,
            confidence_threshold=confidence_threshold
        )
        
        self.intersection_detector_advanced = IntersectionDetectorAdvanced(
            v_fov=0.6,
            min_points=4
        )
        
        # Agregar atributo authority al line follower
        self.authority = 1.0
        
        # Navigator que coordina todo
        self.track_navigator = TrackNavigator(
            line_follower=self,
            sign_detector=self.sign_detector,
            intersection_detector=self.intersection_detector_advanced,
            stoplight_detector=self.stoplight_detector,
            flag_detector=self.flag_detector 
        )
        
        # Selector de modo: True para modo mejorado con YOLO, False para modo original
        self.use_enhanced_mode = self.use_yolo  # Usar modo mejorado solo si YOLO está disponible

        self.get_logger().info(f"Line follower initialized with YOLO: {self.use_yolo}")
        self.get_logger().info(f"Enhanced mode: {self.use_enhanced_mode}")

     # ✨ AGREGAR AQUÍ LA FUNCIÓN DE SATURACIÓN
    
    def enhance_saturation(self, frame, saturation_factor=1.4):
        #Aumentar saturación para colores más vivos
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

            # Aplicar authority si está definido
            if hasattr(self, 'authority'):
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
        #Callback principal con detección de intersecciones y YOLO
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            debug = frame.copy()

            # NO aplicar enhance_saturation al frame principal
            
            if self.use_enhanced_mode and self.use_yolo:
                # ============ MODO MEJORADO CON YOLO Y TRACK NAVIGATOR ============
                # CREAR una copia enhanced SOLO para YOLO, sin modificar el frame original
                enhanced_frame = self.enhance_saturation(frame, saturation_factor=1.4)
                v, w = self.track_navigator.navigate(enhanced_frame, drawing_frame=debug)
                
                # Información de debug mejorada
                cv2.putText(debug, "ENHANCED MODE WITH YOLO", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(debug, f"Signs detected: {len(self.track_navigator.last_signs)}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Mostrar información de las señales detectadas
                if self.track_navigator.last_signs:
                    for i, sign in enumerate(self.track_navigator.last_signs[:3]):  # Máximo 3 señales
                        sign_text = f"{sign.type.name}: {sign.confidence:.2f}"
                        if sign.approx_dist:
                            sign_text += f" ({sign.approx_dist:.2f}m)"
                        cv2.putText(debug, sign_text, (10, 90 + i*20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Mostrar información del estado de navegación
                state_color = {
                    "FOLLOWING": (0, 255, 0),
                    "INTERSECTION_DETECTED": (0, 255, 255), 
                    "CROSSING": (255, 0, 255),
                    "TURNING": (255, 255, 0),
                    "RESUMING": (0, 255, 0)
                }.get(self.track_navigator.state, (255, 255, 255))
                
                cv2.putText(debug, f"Navigation State: {self.track_navigator.state}", 
                        (debug.shape[1] - 400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
                
                if self.track_navigator.turn_command:
                    cv2.putText(debug, f"Next Turn: {self.track_navigator.turn_command.name}", 
                            (debug.shape[1] - 400, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Mostrar timers activos
                if self.track_navigator.state == "CROSSING":
                    cv2.putText(debug, f"Crossing Timer: {self.track_navigator.crossing_timer}", 
                            (debug.shape[1] - 400, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                elif self.track_navigator.state == "TURNING":
                    cv2.putText(debug, f"Turning Timer: {self.track_navigator.turning_timer}", 
                            (debug.shape[1] - 400, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                elif self.track_navigator.state == "RESUMING":
                    cv2.putText(debug, f"Resuming Timer: {self.track_navigator.resuming_timer}", 
                            (debug.shape[1] - 400, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
            else:
                # ============ MODO ORIGINAL SIN YOLO ============
                # Usar frame original SIN modificaciones
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
                    self.intersection_timeout = 30
                elif self.intersection_timeout > 0:
                    self.mode = "INTERSECTION"
                    self.intersection_timeout -= 1
                else:
                    self.mode = "FOLLOWING"
                    if hasattr(self, '_last_mode') and self._last_mode == "INTERSECTION":
                        self.yaw_pid.reset()
                
                # Control según el modo
                if self.mode == "INTERSECTION":
                    v, w = self.stop_at_intersection(frame, self.last_intersection, drawing_frame=debug)
                    
                    cv2.putText(debug, "ORIGINAL INTERSECTION MODE", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(debug, f"Confidence: {self.intersection_confidence}/{self.min_confidence}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(debug, f"Directions: {len(valid_directions)}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(debug, f"Timeout: {self.intersection_timeout}", (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                else:  # FOLLOWING mode
                    v, w = self.follow_line(frame, drawing_frame=debug)
                    
                    cv2.putText(debug, "ORIGINAL LINE FOLLOWING MODE", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(debug, f"v: {v:.2f} m/s", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(debug, f"w: {math.degrees(w):.1f}°/s", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(debug, f"straight_cnt: {self.straight_counter}", (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                self._last_mode = self.mode
            
            # ============ PUBLICAR COMANDOS ============
            twist = Twist()
            twist.linear.x = float(v)
            twist.angular.z = float(w)
            self.cmd_pub.publish(twist)
            
            # Publicar estado mejorado
            status_msg = String()
            if self.use_enhanced_mode and self.use_yolo:
                state = self.track_navigator.state
                turn_cmd = self.track_navigator.turn_command.name if self.track_navigator.turn_command else "NONE"
                status_msg.data = f"YOLO_STATE:{state}, TURN:{turn_cmd}, v={v:.3f}, w={math.degrees(w):.1f}°/s, signs={len(self.track_navigator.last_signs)}"
            else:
                status_msg.data = f"{self.mode}: v={v:.3f}, w={math.degrees(w):.1f}°/s"
            self.status_pub.publish(status_msg)
            
            # Publicar imagen de debug
            debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
            self.debug_pub.publish(debug_msg)
            
            # Log según el modo
            mode_str = "ENHANCED_YOLO" if (self.use_enhanced_mode and self.use_yolo) else self.mode
            if self.get_logger().get_effective_level() <= 20:  # Solo si el nivel de log es INFO o menor
                self.get_logger().info(f"{mode_str}: v={v:.3f}, w={math.degrees(w):.1f}°/s")
            
        except Exception as e:
            self.get_logger().error(f"Error in image processing: {str(e)}")
            # En caso de error, mantener último comando seguro
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerWithYOLONode()
    
    try:
        node.get_logger().info("🚀 PuzzleBot Line Follower with YOLO started")
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Stopping PuzzleBot...")
    finally:
        # Detener el robot antes de cerrar
        stop_cmd = Twist()
        node.cmd_pub.publish(stop_cmd)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()