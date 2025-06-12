#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional
from enum import Enum
from ultralytics import YOLO

# ==================== CLASES Y ENUMS ORIGINALES ====================

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
        """Detecta señales con YOLO y retorna información en formato compatible"""
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
        """Mapea nombres de clases YOLO a tipos de señales del sistema original"""
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
        elif ('yield' in class_name or 'ceda' in class_name or 'give way' in class_name or 'giveway' in class_name):
            return 5  # YIELD
        elif 'work' in class_name or 'construction' in class_name or 'obras' in class_name:
            return 6  # ROAD_WORK
        else:
            return 3  # Default to FORWARD

class SignDetector:
    def __init__(self, get_signs_func=None, confidence_threshold=0.5, ref_height=50):
        self.confidence_threshold = confidence_threshold
        self._get_signs_func = get_signs_func or (lambda frame, drawing_frame=None: ([], [], [], []))
        self.ref_height = ref_height
        self.chain_length = 8
        self.max_chain_gap = 2
        history_length = max(4, self.chain_length + self.max_chain_gap)
        self._sign_histories = {sign_type: deque(maxlen=history_length) for sign_type in SignType}

    def get_confirmed_signs_nb(self, frame, drawing_frame=None):
        return self.get_confirmed_signs(frame, drawing_frame)

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

# ==================== MENSAJE PERSONALIZADO (SIMULADO) ====================

class DetectedSign:
    def __init__(self):
        self.type = 0
        self.box = [0.0, 0.0, 0.0, 0.0]
        self.confidence = 0.0
        self.approx_dist = 0.0
        self.timestamp = 0.0

class DetectedSigns:
    def __init__(self):
        self.header = Header()
        self.signs = []

# ==================== NODO YOLO DETECTOR ====================

class YOLODetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')
        
        # Parámetros
        self.declare_parameter('model_path', 'traffic_signs_dataset/best.pt')
        self.declare_parameter('confidence_threshold', 0.6)
        self.declare_parameter('use_yolo', True)
        
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.use_yolo = self.get_parameter('use_yolo').get_parameter_value().bool_value
        
        # Inicializar YOLO
        self.yolo_detector = None
        if self.use_yolo:
            self.yolo_detector = YOLOSignDetector(model_path, confidence_threshold)
            if self.yolo_detector.model is not None:
                self.get_logger().info('✅ YOLO detector initialized successfully')
            else:
                self.get_logger().error('❌ Failed to initialize YOLO detector')
                self.use_yolo = False
        
        # Función de detección para SignDetector
        def get_signs_func(frame, drawing_frame=None):
            if self.yolo_detector:
                return self.yolo_detector.detect_signs(frame)
            return [], [], [], []
        
        # Inicializar SignDetector
        self.sign_detector = SignDetector(
            get_signs_func=get_signs_func,
            confidence_threshold=confidence_threshold
        )
        
        # ROS2 setup
        self.bridge = CvBridge()
        
        # Publishers
        self.signs_pub = self.create_publisher(DetectedSigns, '/detected_signs', 10)
        self.debug_pub = self.create_publisher(Image, '/yolo_debug_image', 10)
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10)
        
        self.get_logger().info(f'YOLO Detector Node initialized. YOLO enabled: {self.use_yolo}')
    
    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            debug_frame = frame.copy()
            
            # Detectar señales confirmadas
            confirmed_signs = self.sign_detector.get_confirmed_signs_nb(frame, drawing_frame=debug_frame)
            
            # Crear mensaje de señales detectadas
            signs_msg = DetectedSigns()
            signs_msg.header = msg.header
            signs_msg.header.frame_id = "camera_frame"
            
            for sign in confirmed_signs:
                sign_msg = DetectedSign()
                sign_msg.type = sign.type.value
                sign_msg.box = sign.box.tolist() if isinstance(sign.box, np.ndarray) else sign.box
                sign_msg.confidence = sign.confidence
                sign_msg.approx_dist = sign.approx_dist if sign.approx_dist is not None else 0.0
                sign_msg.timestamp = sign.timestamp if sign.timestamp is not None else 0.0
                signs_msg.signs.append(sign_msg)
            
            # Publicar señales detectadas
            self.signs_pub.publish(signs_msg)
            
            # Agregar información de debug
            cv2.putText(debug_frame, f"YOLO Signs: {len(confirmed_signs)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if confirmed_signs:
                for i, sign in enumerate(confirmed_signs):
                    text = f"{sign.type.name}: {sign.confidence:.2f}"
                    if sign.approx_dist:
                        text += f" ({sign.approx_dist:.2f}m)"
                    cv2.putText(debug_frame, text, (10, 60 + i*20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Publicar imagen de debug
            debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
            self.debug_pub.publish(debug_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error in YOLO detection: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = YOLODetectorNode()
    
    try:
        node.get_logger().info("🚀 YOLO Detector Node started")
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Stopping YOLO Detector Node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()