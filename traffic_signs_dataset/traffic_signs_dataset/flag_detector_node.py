#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header, Float32, Bool
from cv_bridge import CvBridge
import cv2
import numpy as np

# ==================== CLASE DETECTOR DE BANDERA ORIGINAL ====================

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

# ==================== MENSAJE PERSONALIZADO (SIMULADO) ====================

class FlagData:
    def __init__(self):
        self.header = Header()
        self.detected = False
        self.distance = 0.0
        self.end_reached = False

# ==================== NODO DETECTOR DE BANDERA ====================

class FlagDetectorNode(Node):
    def __init__(self):
        super().__init__('flag_detector_node')
        
        # Parámetros
        self.declare_parameter('distance_threshold', 0.40)
        self.declare_parameter('pattern_size_x', 4)
        self.declare_parameter('pattern_size_y', 3)
        self.declare_parameter('square_size', 0.025)
        
        dist_thres = self.get_parameter('distance_threshold').get_parameter_value().double_value
        pattern_x = self.get_parameter('pattern_size_x').get_parameter_value().integer_value
        pattern_y = self.get_parameter('pattern_size_y').get_parameter_value().integer_value
        square_size = self.get_parameter('square_size').get_parameter_value().double_value
        
        # Inicializar detector
        self.flag_detector = FlagDetector(dist_thres=dist_thres)
        
        # Actualizar parámetros si se proporcionaron
        self.flag_detector.pattern_size = (pattern_x, pattern_y)
        self.flag_detector.square_size = square_size
        
        # ROS2 setup
        self.bridge = CvBridge()
        
        # Publishers
        self.flag_pub = self.create_publisher(FlagData, '/flag_data', 10)
        self.distance_pub = self.create_publisher(Float32, '/flag_distance', 10)
        self.end_reached_pub = self.create_publisher(Bool, '/end_reached', 10)
        self.debug_pub = self.create_publisher(Image, '/flag_debug_image', 10)
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10)
        
        self.get_logger().info(f'Flag Detector Node initialized. Distance threshold: {dist_thres}m')
    
    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            debug_frame = frame.copy()
            
            # Detectar bandera
            flag_distance = self.flag_detector.get_flag_distance_nb(frame, drawing_frame=debug_frame)
            
            # Determinar si se detectó bandera
            flag_detected = flag_distance is not None
            
            # Verificar si se alcanzó el final
            if flag_detected and flag_distance <= self.flag_detector.dist_thres:
                self.flag_detector.end_reached = True
            
            # Crear mensaje completo de bandera
            flag_msg = FlagData()
            flag_msg.header = msg.header
            flag_msg.header.frame_id = "camera_frame"
            flag_msg.detected = flag_detected
            flag_msg.distance = flag_distance if flag_distance is not None else 0.0
            flag_msg.end_reached = self.flag_detector.end_reached
            
            # Crear mensajes simples
            distance_msg = Float32()
            distance_msg.data = flag_distance if flag_distance is not None else -1.0
            
            end_msg = Bool()
            end_msg.data = self.flag_detector.end_reached
            
            # Publicar mensajes
            self.flag_pub.publish(flag_msg)
            self.distance_pub.publish(distance_msg)
            self.end_reached_pub.publish(end_msg)
            
            # Agregar información de debug
            if flag_detected:
                color = (0, 255, 0) if flag_distance <= self.flag_detector.dist_thres else (0, 255, 255)
                status = "END REACHED!" if self.flag_detector.end_reached else "FLAG DETECTED"
                cv2.putText(debug_frame, f"{status}: {flag_distance:.2f}m", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(debug_frame, f"Threshold: {self.flag_detector.dist_thres:.2f}m", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(debug_frame, "No flag detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Mostrar estado del final
            if self.flag_detector.end_reached:
                cv2.putText(debug_frame, "🏁 RACE FINISHED! 🏁", 
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            
            # Mostrar parámetros del checkerboard
            cv2.putText(debug_frame, f"Pattern: {self.flag_detector.pattern_size}", 
                       (10, debug_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(debug_frame, f"Square size: {self.flag_detector.square_size}m", 
                       (10, debug_frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(debug_frame, f"Dist threshold: {self.flag_detector.dist_thres}m", 
                       (10, debug_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Publicar imagen de debug
            debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
            self.debug_pub.publish(debug_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error in flag detection: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = FlagDetectorNode()
    
    try:
        node.get_logger().info("🚀 Flag Detector Node started")
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Stopping Flag Detector Node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()