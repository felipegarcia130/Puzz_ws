#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header, Int32
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import deque

# ==================== CLASE DETECTOR DE SEMÁFORO ORIGINAL ====================

class StoplightDetector:
    def __init__(self,
        v_fov=0.5,
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
        lower_red1 = np.array([0, 120, 100])    # Saturación y brillo más altos
        upper_red1 = np.array([8, 255, 255])    # Tono más específico
        lower_red2 = np.array([172, 120, 100])  # Saturación y brillo más altos
        upper_red2 = np.array([180, 255, 255])

        # Amarillo - rangos MUY específicos para evitar traslape con rojo y verde
        lower_yellow1 = np.array([18, 100, 150])   # Amarillo puro, saturación alta
        upper_yellow1 = np.array([25, 255, 255])   # Rango muy estrecho
        
        # Amarillo claro pero MUY restrictivo
        lower_yellow2 = np.array([26, 30, 200])    # Solo amarillos claros muy específicos
        upper_yellow2 = np.array([32, 80, 255])    # Evita traslape con verde (40+)

        # Crear máscaras
        mask_green = cv2.inRange(hsv_proc, lower_green, upper_green)
        mask_red1 = cv2.inRange(hsv_proc, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv_proc, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        # Combinar ambos rangos de amarillo
        mask_yellow1 = cv2.inRange(hsv_proc, lower_yellow1, upper_yellow1)
        mask_yellow2 = cv2.inRange(hsv_proc, lower_yellow2, upper_yellow2)
        mask_yellow = cv2.bitwise_or(mask_yellow1, mask_yellow2)

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

        # Debug: Número de contornos encontrados (opcional)
        if len(contours_green) > 0 or len(contours_red) > 0 or len(contours_yellow) > 0:
            print(f'🔍 Contornos encontrados - Verde: {len(contours_green)}, Rojo: {len(contours_red)}, Amarillo: {len(contours_yellow)}')

        # Detectar qué colores están presentes
        colors_detected = []

        # 🟡 AMARILLO - Detección con filtros
        if contours_yellow:
            largest_yellow = max(contours_yellow, key=cv2.contourArea)
            yellow_area = cv2.contourArea(largest_yellow)
            
            if yellow_area > 200:  # Área mínima
                perimeter = cv2.arcLength(largest_yellow, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * yellow_area / (perimeter * perimeter)
                    
                    # Solo aceptar formas circulares (semáforo)
                    if circularity > 0.3:  # Menos restrictivo que el rojo
                        yellow_detected = True
                        colors_detected.append(f"AMARILLO (área: {yellow_area:.0f})")
                        if drawing_frame is not None:
                            cv2.drawContours(drawing_frame, [largest_yellow], -1, (0, 255, 255), 3)
                            cv2.putText(drawing_frame, f"YELLOW: {yellow_area:.0f}", 
                                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 🟥 ROJO - Detección con filtros estrictos
        if contours_red:
            largest_red = max(contours_red, key=cv2.contourArea)
            red_area = cv2.contourArea(largest_red)
            
            if red_area > 100:  # Filtro inicial de área
                if red_area > 200:  # Área mínima alta para semáforos
                    # Verificar que el contorno sea más o menos circular/cuadrado
                    perimeter = cv2.arcLength(largest_red, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * red_area / (perimeter * perimeter)
                        
                        # Solo aceptar formas muy circulares (semáforo)
                        if circularity > 0.4:  # Más restrictivo
                            red_detected = True
                            colors_detected.append(f"ROJO (área: {red_area:.0f})")
                            if drawing_frame is not None:
                                cv2.drawContours(drawing_frame, [largest_red], -1, (0, 0, 255), 3)
                                cv2.putText(drawing_frame, f"RED: {red_area:.0f}", 
                                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 🟢 VERDE - Detección
        if contours_green:
            largest_green = max(contours_green, key=cv2.contourArea)
            green_area = cv2.contourArea(largest_green)
            
            if green_area > 800:  # Área mínima para verde (del código original)
                green_detected = True
                colors_detected.append(f"VERDE (área: {green_area:.0f})")
                if drawing_frame is not None:
                    cv2.drawContours(drawing_frame, [largest_green], -1, (0, 255, 0), 3)
                    cv2.putText(drawing_frame, f"GREEN: {green_area:.0f}", 
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Imprimir colores detectados (opcional)
        if colors_detected:
            print(f'🎯 Colores detectados: {", ".join(colors_detected)}')
        else:
            # Solo mostrar cada ciertos frames para no saturar
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

        # Retornar estado confirmado
        if confirmed_red:
            return 0  # Red
        elif confirmed_yellow:
            return 1  # Yellow
        elif confirmed_green:
            return 2  # Green
        return None  # No detection

# ==================== MENSAJE PERSONALIZADO (SIMULADO) ====================

class StoplightState:
    def __init__(self):
        self.header = Header()
        self.state = -1  # -1: unknown, 0: red, 1: yellow, 2: green
        self.red_detected = False
        self.yellow_detected = False
        self.green_detected = False

# ==================== NODO DETECTOR DE SEMÁFORO ====================

class StoplightDetectorNode(Node):
    def __init__(self):
        super().__init__('stoplight_detector_node')
        
        # Parámetros
        self.declare_parameter('v_fov', 0.5)
        self.declare_parameter('chain_length', 4)
        self.declare_parameter('max_chain_gap', 1)
        self.declare_parameter('history_len', 5)
        self.declare_parameter('show_unconfirmed', False)
        self.declare_parameter('saturation_factor', 1.4)
        
        v_fov = self.get_parameter('v_fov').get_parameter_value().double_value
        chain_length = self.get_parameter('chain_length').get_parameter_value().integer_value
        max_chain_gap = self.get_parameter('max_chain_gap').get_parameter_value().integer_value
        history_len = self.get_parameter('history_len').get_parameter_value().integer_value
        show_unconfirmed = self.get_parameter('show_unconfirmed').get_parameter_value().bool_value
        
        # Inicializar detector
        self.stoplight_detector = StoplightDetector(
            v_fov=v_fov,
            chain_length=chain_length,
            max_chain_gap=max_chain_gap,
            history_len=history_len,
            show_unconfirmed=show_unconfirmed
        )
        
        # ROS2 setup
        self.bridge = CvBridge()
        
        # Publishers
        self.stoplight_pub = self.create_publisher(StoplightState, '/stoplight_state', 10)
        self.stoplight_simple_pub = self.create_publisher(Int32, '/stoplight_simple', 10)
        self.debug_pub = self.create_publisher(Image, '/stoplight_debug_image', 10)
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10)
        
        self.get_logger().info('Stoplight Detector Node initialized')
    
    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            debug_frame = frame.copy()
            
            # Detectar estado del semáforo
            stoplight_state = self.stoplight_detector.identify_stoplight(frame, drawing_frame=debug_frame)
            
            # Obtener detecciones individuales para el mensaje completo
            red_detected, yellow_detected, green_detected = self.stoplight_detector.classify_stoplight_colors(frame)
            
            # Crear mensaje completo de estado
            stoplight_msg = StoplightState()
            stoplight_msg.header = msg.header
            stoplight_msg.header.frame_id = "camera_frame"
            stoplight_msg.state = stoplight_state if stoplight_state is not None else -1
            stoplight_msg.red_detected = red_detected
            stoplight_msg.yellow_detected = yellow_detected
            stoplight_msg.green_detected = green_detected
            
            # Crear mensaje simple
            simple_msg = Int32()
            simple_msg.data = stoplight_state if stoplight_state is not None else -1
            
            # Publicar mensajes
            self.stoplight_pub.publish(stoplight_msg)
            self.stoplight_simple_pub.publish(simple_msg)
            
            # Agregar información de debug
            state_names = {-1: "UNKNOWN", 0: "RED", 1: "YELLOW", 2: "GREEN"}
            state_colors = {-1: (128, 128, 128), 0: (0, 0, 255), 1: (0, 255, 255), 2: (0, 255, 0)}
            
            state_name = state_names.get(stoplight_msg.state, "UNKNOWN")
            state_color = state_colors.get(stoplight_msg.state, (128, 128, 128))
            
            cv2.putText(debug_frame, f"Stoplight: {state_name}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
            
            # Mostrar detecciones individuales
            detection_text = f"Detections - R:{red_detected} Y:{yellow_detected} G:{green_detected}"
            cv2.putText(debug_frame, detection_text, 
                       (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Mostrar historiales
            red_conf = sum(self.stoplight_detector.red_history)
            yellow_conf = sum(self.stoplight_detector.yellow_history)
            green_conf = sum(self.stoplight_detector.green_history)
            
            cv2.putText(debug_frame, f"Confidence - R:{red_conf} Y:{yellow_conf} G:{green_conf}", 
                       (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Publicar imagen de debug
            debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
            self.debug_pub.publish(debug_msg)
            
            # Incrementar contador de frames
            self.stoplight_detector.frame_count += 1
            
        except Exception as e:
            self.get_logger().error(f'Error in stoplight detection: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = StoplightDetectorNode()
    
    try:
        node.get_logger().info("🚀 Stoplight Detector Node started")
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Stopping Stoplight Detector Node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header, Int32
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import deque

# ==================== CLASE DETECTOR DE SEMÁFORO ORIGINAL ====================

class StoplightDetector:
    def __init__(self,
        v_fov=0.5,
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
        lower_red1 = np.array([0, 120, 100])    # Saturación y brillo más altos
        upper_red1 = np.array([8, 255, 255])    # Tono más específico
        lower_red2 = np.array([172, 120, 100])  # Saturación y brillo más altos
        upper_red2 = np.array([180, 255, 255])

        # Amarillo - rangos MUY específicos para evitar traslape con rojo y verde
        lower_yellow1 = np.array([18, 100, 150])   # Amarillo puro, saturación alta
        upper_yellow1 = np.array([25, 255, 255])   # Rango muy estrecho
        
        # Amarillo claro pero MUY restrictivo
        lower_yellow2 = np.array([26, 30, 200])    # Solo amarillos claros muy específicos
        upper_yellow2 = np.array([32, 80, 255])    # Evita traslape con verde (40+)

        # Crear máscaras
        mask_green = cv2.inRange(hsv_proc, lower_green, upper_green)
        mask_red1 = cv2.inRange(hsv_proc, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv_proc, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        # Combinar ambos rangos de amarillo
        mask_yellow1 = cv2.inRange(hsv_proc, lower_yellow1, upper_yellow1)
        mask_yellow2 = cv2.inRange(hsv_proc, lower_yellow2, upper_yellow2)
        mask_yellow = cv2.bitwise_or(mask_yellow1, mask_yellow2)

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

        # Debug: Número de contornos encontrados (opcional)
        if len(contours_green) > 0 or len(contours_red) > 0 or len(contours_yellow) > 0:
            print(f'🔍 Contornos encontrados - Verde: {len(contours_green)}, Rojo: {len(contours_red)}, Amarillo: {len(contours_yellow)}')

        # Detectar qué colores están presentes
        colors_detected = []

        # 🟡 AMARILLO - Detección con filtros
        if contours_yellow:
            largest_yellow = max(contours_yellow, key=cv2.contourArea)
            yellow_area = cv2.contourArea(largest_yellow)
            
            if yellow_area > 200:  # Área mínima
                perimeter = cv2.arcLength(largest_yellow, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * yellow_area / (perimeter * perimeter)
                    
                    # Solo aceptar formas circulares (semáforo)
                    if circularity > 0.3:  # Menos restrictivo que el rojo
                        yellow_detected = True
                        colors_detected.append(f"AMARILLO (área: {yellow_area:.0f})")
                        if drawing_frame is not None:
                            cv2.drawContours(drawing_frame, [largest_yellow], -1, (0, 255, 255), 3)
                            cv2.putText(drawing_frame, f"YELLOW: {yellow_area:.0f}", 
                                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 🟥 ROJO - Detección con filtros estrictos
        if contours_red:
            largest_red = max(contours_red, key=cv2.contourArea)
            red_area = cv2.contourArea(largest_red)
            
            if red_area > 100:  # Filtro inicial de área
                if red_area > 200:  # Área mínima alta para semáforos
                    # Verificar que el contorno sea más o menos circular/cuadrado
                    perimeter = cv2.arcLength(largest_red, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * red_area / (perimeter * perimeter)
                        
                        # Solo aceptar formas muy circulares (semáforo)
                        if circularity > 0.4:  # Más restrictivo
                            red_detected = True
                            colors_detected.append(f"ROJO (área: {red_area:.0f})")
                            if drawing_frame is not None:
                                cv2.drawContours(drawing_frame, [largest_red], -1, (0, 0, 255), 3)
                                cv2.putText(drawing_frame, f"RED: {red_area:.0f}", 
                                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 🟢 VERDE - Detección
        if contours_green:
            largest_green = max(contours_green, key=cv2.contourArea)
            green_area = cv2.contourArea(largest_green)
            
            if green_area > 800:  # Área mínima para verde (del código original)
                green_detected = True
                colors_detected.append(f"VERDE (área: {green_area:.0f})")
                if drawing_frame is not None:
                    cv2.drawContours(drawing_frame, [largest_green], -1, (0, 255, 0), 3)
                    cv2.putText(drawing_frame, f"GREEN: {green_area:.0f}", 
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Imprimir colores detectados (opcional)
        if colors_detected:
            print(f'🎯 Colores detectados: {", ".join(colors_detected)}')
        else:
            # Solo mostrar cada ciertos frames para no saturar
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

        # Retornar estado confirmado
        if confirmed_red:
            return 0  # Red
        elif confirmed_yellow:
            return 1  # Yellow
        elif confirmed_green:
            return 2  # Green
        return None  # No detection

# ==================== MENSAJE PERSONALIZADO (SIMULADO) ====================

class StoplightState:
    def __init__(self):
        self.header = Header()
        self.state = -1  # -1: unknown, 0: red, 1: yellow, 2: green
        self.red_detected = False
        self.yellow_detected = False
        self.green_detected = False

# ==================== NODO DETECTOR DE SEMÁFORO ====================

class StoplightDetectorNode(Node):
    def __init__(self):
        super().__init__('stoplight_detector_node')
        
        # Parámetros
        self.declare_parameter('v_fov', 0.5)
        self.declare_parameter('chain_length', 4)
        self.declare_parameter('max_chain_gap', 1)
        self.declare_parameter('history_len', 5)
        self.declare_parameter('show_unconfirmed', False)
        self.declare_parameter('saturation_factor', 1.4)
        
        v_fov = self.get_parameter('v_fov').get_parameter_value().double_value
        chain_length = self.get_parameter('chain_length').get_parameter_value().integer_value
        max_chain_gap = self.get_parameter('max_chain_gap').get_parameter_value().integer_value
        history_len = self.get_parameter('history_len').get_parameter_value().integer_value
        show_unconfirmed = self.get_parameter('show_unconfirmed').get_parameter_value().bool_value
        
        # Inicializar detector
        self.stoplight_detector = StoplightDetector(
            v_fov=v_fov,
            chain_length=chain_length,
            max_chain_gap=max_chain_gap,
            history_len=history_len,
            show_unconfirmed=show_unconfirmed
        )
        
        # ROS2 setup
        self.bridge = CvBridge()
        
        # Publishers
        self.stoplight_pub = self.create_publisher(StoplightState, '/stoplight_state', 10)
        self.stoplight_simple_pub = self.create_publisher(Int32, '/stoplight_simple', 10)
        self.debug_pub = self.create_publisher(Image, '/stoplight_debug_image', 10)
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10)
        
        self.get_logger().info('Stoplight Detector Node initialized')
    
    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            debug_frame = frame.copy()
            
            # Detectar estado del semáforo
            stoplight_state = self.stoplight_detector.identify_stoplight(frame, drawing_frame=debug_frame)
            
            # Obtener detecciones individuales para el mensaje completo
            red_detected, yellow_detected, green_detected = self.stoplight_detector.classify_stoplight_colors(frame)
            
            # Crear mensaje completo de estado
            stoplight_msg = StoplightState()
            stoplight_msg.header = msg.header
            stoplight_msg.header.frame_id = "camera_frame"
            stoplight_msg.state = stoplight_state if stoplight_state is not None else -1
            stoplight_msg.red_detected = red_detected
            stoplight_msg.yellow_detected = yellow_detected
            stoplight_msg.green_detected = green_detected
            
            # Crear mensaje simple
            simple_msg = Int32()
            simple_msg.data = stoplight_state if stoplight_state is not None else -1
            
            # Publicar mensajes
            self.stoplight_pub.publish(stoplight_msg)
            self.stoplight_simple_pub.publish(simple_msg)
            
            # Agregar información de debug
            state_names = {-1: "UNKNOWN", 0: "RED", 1: "YELLOW", 2: "GREEN"}
            state_colors = {-1: (128, 128, 128), 0: (0, 0, 255), 1: (0, 255, 255), 2: (0, 255, 0)}
            
            state_name = state_names.get(stoplight_msg.state, "UNKNOWN")
            state_color = state_colors.get(stoplight_msg.state, (128, 128, 128))
            
            cv2.putText(debug_frame, f"Stoplight: {state_name}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
            
            # Mostrar detecciones individuales
            detection_text = f"Detections - R:{red_detected} Y:{yellow_detected} G:{green_detected}"
            cv2.putText(debug_frame, detection_text, 
                       (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Mostrar historiales
            red_conf = sum(self.stoplight_detector.red_history)
            yellow_conf = sum(self.stoplight_detector.yellow_history)
            green_conf = sum(self.stoplight_detector.green_history)
            
            cv2.putText(debug_frame, f"Confidence - R:{red_conf} Y:{yellow_conf} G:{green_conf}", 
                       (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Publicar imagen de debug
            debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
            self.debug_pub.publish(debug_msg)
            
            # Incrementar contador de frames
            self.stoplight_detector.frame_count += 1
            
        except Exception as e:
            self.get_logger().error(f'Error in stoplight detection: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = StoplightDetectorNode()
    
    try:
        node.get_logger().info("🚀 Stoplight Detector Node started")
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Stopping Stoplight Detector Node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()