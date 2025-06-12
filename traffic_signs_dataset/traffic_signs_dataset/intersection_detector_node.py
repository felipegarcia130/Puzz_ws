#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from itertools import combinations

# ==================== FUNCIONES ORIGINALES DE INTERSECCIÓN ====================

def group_dotted_lines_simple(points, min_inliers=4, dist_threshold=3.0, distance_ratio=2.5):
    """Groups points into dotted lines"""
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

# ==================== CLASE DETECTOR AVANZADO ORIGINAL ====================

class IntersectionDetectorAdvanced:
    def __init__(self, v_fov=0.55, min_points=5, max_yaw=30.0, max_thr=0.15):
        self.v_fov = v_fov
        self.morph_kernel = np.ones((3, 3), np.uint8)
        self.erode_iterations = 3
        self.dilate_iterations = 2
        self.max_aspect_ratio = 10.0
        self.min_area = 20
        self.ep = 0.035
        self.min_points = min_points
        self.yaw_threshold = 5.0

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

# ==================== MENSAJE PERSONALIZADO (SIMULADO) ====================

class IntersectionData:
    def __init__(self):
        self.header = Header()
        self.detected = False
        self.center = Point()
        self.angle = 0.0
        self.confidence = 0
        self.back_detected = False
        self.left_detected = False
        self.right_detected = False
        self.front_detected = False

# ==================== NODO DETECTOR DE INTERSECCIÓN ====================

class IntersectionDetectorNode(Node):
    def __init__(self):
        super().__init__('intersection_detector_node')
        
        # Parámetros
        self.declare_parameter('v_fov', 0.6)
        self.declare_parameter('min_points', 4)
        self.declare_parameter('min_confidence', 3)
        
        v_fov = self.get_parameter('v_fov').get_parameter_value().double_value
        min_points = self.get_parameter('min_points').get_parameter_value().integer_value
        self.min_confidence = self.get_parameter('min_confidence').get_parameter_value().integer_value
        
        # Inicializar detector
        self.intersection_detector = IntersectionDetectorAdvanced(
            v_fov=v_fov,
            min_points=min_points
        )
        
        # Variables de estado
        self.intersection_confidence = 0
        self.last_intersection = None
        
        # ROS2 setup
        self.bridge = CvBridge()
        
        # Publishers
        self.intersection_pub = self.create_publisher(IntersectionData, '/intersection_data', 10)
        self.debug_pub = self.create_publisher(Image, '/intersection_debug_image', 10)
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10)
        
        self.get_logger().info('Intersection Detector Node initialized')
    
    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            debug_frame = frame.copy()
            
            # Detectar intersección usando el método original
            intersection_directions = identify_intersection(frame, drawing_frame=debug_frame)
            back, left, right, front = intersection_directions
            
            # Detectar usando el método avanzado
            advanced_intersection = self.intersection_detector.find_intersection(frame, drawing_frame=debug_frame)
            
            # Contar direcciones válidas
            valid_directions = [d for d in intersection_directions if d is not None]
            intersection_detected = len(valid_directions) >= 2
            
            # Sistema de confianza
            if intersection_detected:
                self.intersection_confidence = min(self.intersection_confidence + 1, self.min_confidence + 2)
                self.last_intersection = intersection_directions
            else:
                self.intersection_confidence = max(self.intersection_confidence - 1, 0)
            
            # Crear mensaje de intersección
            intersection_msg = IntersectionData()
            intersection_msg.header = msg.header
            intersection_msg.header.frame_id = "camera_frame"
            intersection_msg.detected = self.intersection_confidence >= self.min_confidence
            intersection_msg.confidence = self.intersection_confidence
            intersection_msg.back_detected = back is not None
            intersection_msg.left_detected = left is not None
            intersection_msg.right_detected = right is not None
            intersection_msg.front_detected = front is not None
            
            # Si hay intersección avanzada detectada, usar esos datos
            if advanced_intersection is not None:
                line, center, angle = advanced_intersection
                intersection_msg.center.x = float(center[0])
                intersection_msg.center.y = float(center[1])
                intersection_msg.center.z = 0.0
                intersection_msg.angle = float(angle)
            
            # Publicar datos de intersección
            self.intersection_pub.publish(intersection_msg)
            
            # Agregar información de debug
            cv2.putText(debug_frame, f"Intersection: {intersection_detected}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(debug_frame, f"Confidence: {self.intersection_confidence}/{self.min_confidence}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(debug_frame, f"Directions: {len(valid_directions)}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if advanced_intersection:
                line, center, angle = advanced_intersection
                cv2.putText(debug_frame, f"Advanced: Center=({center[0]}, {center[1]}) Angle={angle:.1f}°", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
            # Publicar imagen de debug
            debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
            self.debug_pub.publish(debug_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error in intersection detection: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = IntersectionDetectorNode()
    
    try:
        node.get_logger().info("🚀 Intersection Detector Node started")
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Stopping Intersection Detector Node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()