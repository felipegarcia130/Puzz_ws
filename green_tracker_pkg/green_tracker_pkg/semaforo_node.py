import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class Semaforo(Node):
    def __init__(self):
        super().__init__('semaforo_node')
        self.bridge = CvBridge()

        self.mission_pub = self.create_publisher(Bool, '/mission_control', 10)
        self.slow_down_pub = self.create_publisher(Bool, '/slow_down', 10)
        self.mask_pub = self.create_publisher(Image, '/mask_debug', 10)
        self.debug_pub = self.create_publisher(Image, '/debug_image', 10)

        self.subscription = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.mission_started = False
        self.waiting_for_green = False
        
        # Debug: Contador de frames procesados
        self.frame_count = 0
        
        # Mensaje de inicio
        self.get_logger().info('🚀 Nodo Semáforo iniciado correctamente')

    def image_callback(self, msg):
        self.frame_count += 1
        
        # Debug: Confirmar que se están recibiendo imágenes
        if self.frame_count % 30 == 0:  # Cada 30 frames (aprox 1 seg)
            self.get_logger().info(f'📸 Procesando frame #{self.frame_count}')
        
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Debug: Tamaño de la imagen
            height, width = frame.shape[:2]
            if self.frame_count == 1:
                self.get_logger().info(f'📐 Resolución de imagen: {width}x{height}')

            # Rangos HSV - EXACTOS del código que funciona
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
            mask_green = cv2.inRange(hsv, lower_green, upper_green)
            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            
            # Combinar ambos rangos de amarillo
            mask_yellow1 = cv2.inRange(hsv, lower_yellow1, upper_yellow1)
            mask_yellow2 = cv2.inRange(hsv, lower_yellow2, upper_yellow2)
            mask_yellow = cv2.bitwise_or(mask_yellow1, mask_yellow2)

            # Operaciones morfológicas
            kernel = np.ones((5, 5), np.uint8)
            mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
            mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
            mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)

            # Debug: Contar píxeles de cada color
            green_pixels = cv2.countNonZero(mask_green)
            red_pixels = cv2.countNonZero(mask_red)
            yellow_pixels = cv2.countNonZero(mask_yellow)
            
            # Debug: Mostrar SIEMPRE los píxeles rojos para debug
            if self.frame_count % 30 == 0:  # Cada segundo
                self.get_logger().info(f'🔴 DEBUG Rojo - Píxeles: {red_pixels} | Verde: {green_pixels}')
            
            # Solo mostrar cuando hay píxeles detectados de otros colores
            if green_pixels > 0:
                self.get_logger().info(f'🎨 Píxeles detectados - Verde: {green_pixels}, Rojo: {red_pixels}, Amarillo: {yellow_pixels}')

            # Publicar máscara para debug
            self.mask_pub.publish(self.bridge.cv2_to_imgmsg(mask_green, encoding='mono8'))

            # Buscar contornos
            contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Debug: Número de contornos encontrados
            if len(contours_green) > 0 or len(contours_red) > 0 or len(contours_yellow) > 0:
                self.get_logger().info(f'🔍 Contornos encontrados - Verde: {len(contours_green)}, Rojo: {len(contours_red)}, Amarillo: {len(contours_yellow)}')

            # Detectar qué colores están presentes
            colors_detected = []

            # 🟡 Amarillo detectado
            if contours_yellow:
                largest_yellow = max(contours_yellow, key=cv2.contourArea)
                yellow_area = cv2.contourArea(largest_yellow)
                self.get_logger().info(f'🟡 Contorno amarillo detectado | Área: {yellow_area:.2f}')
                
                if yellow_area > 200:  # Área mínima
                    # Agregar filtros de forma como en el rojo
                    perimeter = cv2.arcLength(largest_yellow, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * yellow_area / (perimeter * perimeter)
                        
                        # Solo aceptar formas circulares (semáforo)
                        if circularity > 0.3:  # Menos restrictivo que el rojo
                            colors_detected.append(f"AMARILLO (área: {yellow_area:.0f})")
                            self.get_logger().info(f'🟡 SEMÁFORO AMARILLO detectado | Área: {yellow_area:.2f} | Circularidad: {circularity:.2f}')
                            
                            # Dibujar contorno amarillo para debug
                            cv2.drawContours(frame, [largest_yellow], -1, (0, 255, 255), 3)
                        else:
                            self.get_logger().info(f'🔶 Objeto amarillo rechazado (forma irregular) | Área: {yellow_area:.2f} | Circularidad: {circularity:.2f}')
                    else:
                        self.get_logger().info(f'🔶 Contorno amarillo inválido | Área: {yellow_area:.2f}')

            # 🟥 Rojo detectado
            if contours_red:
                largest_red = max(contours_red, key=cv2.contourArea)
                red_area = cv2.contourArea(largest_red)
                
                # Solo procesar si el área es significativa (evitar ruido)
                if red_area > 100:  # Filtro inicial de área
                    self.get_logger().info(f'🟥 Contorno rojo detectado | Área: {red_area:.2f}')
                    
                    # Filtros estrictos para detectar solo semáforo rojo real
                    if red_area > 200:  # Área mínima alta para semáforos
                        # Verificar que el contorno sea más o menos circular/cuadrado
                        perimeter = cv2.arcLength(largest_red, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * red_area / (perimeter * perimeter)
                            
                            # Solo aceptar formas muy circulares (semáforo)
                            if circularity > 0.4:  # Más restrictivo
                                colors_detected.append(f"ROJO (área: {red_area:.0f})")
                                self.get_logger().info(f'🟥 SEMÁFORO ROJO detectado | Área: {red_area:.2f} | Circularidad: {circularity:.2f}')
                                self.mission_pub.publish(Bool(data=False))
                                self.slow_down_pub.publish(Bool(data=False))
                                self.mission_started = False
                                self.waiting_for_green = True
                                
                                # Dibujar contorno rojo
                                cv2.drawContours(frame, [largest_red], -1, (0, 0, 255), 3)
                            else:
                                self.get_logger().info(f'🔶 Objeto rojo rechazado (forma irregular) | Área: {red_area:.2f} | Circularidad: {circularity:.2f}')
                        else:
                            self.get_logger().info(f'🔶 Contorno rojo inválido | Área: {red_area:.2f}')
                    else:
                        self.get_logger().info(f'🔶 Objeto rojo muy pequeño (ignorado) | Área: {red_area:.2f}')

            # 🟢 Verde detectado
            if contours_green:
                largest_green = max(contours_green, key=cv2.contourArea)
                green_area = cv2.contourArea(largest_green)
                self.get_logger().info(f'🟢 Contorno verde detectado | Área: {green_area:.2f}')

                if green_area > 800:
                    colors_detected.append(f"VERDE (área: {green_area:.0f})")
                    self.get_logger().info(f'🟢 Verde detectado | Área: {green_area:.2f}')
                    
                    # 🟢 Verde → reanudar si estaba esperando
                    if self.waiting_for_green:
                        self.get_logger().info('🟢 Verde detectado: REANUDANDO MISIÓN')
                        self.mission_pub.publish(Bool(data=True))
                        self.slow_down_pub.publish(Bool(data=False))
                        self.mission_started = True
                        self.waiting_for_green = False
                    elif not self.mission_started:
                        self.get_logger().info(f'🟢 Verde detectado: INICIANDO MISIÓN')
                        self.mission_pub.publish(Bool(data=True))
                        self.slow_down_pub.publish(Bool(data=False))
                        self.mission_started = True

                    # Dibujar el contorno
                    M = cv2.moments(largest_green)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.drawContours(frame, [largest_green], -1, (0, 255, 0), 3)
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # 🟨 Amarillo → bajar velocidad si ya está en misión
            if contours_yellow and self.mission_started:
                largest_yellow = max(contours_yellow, key=cv2.contourArea)
                yellow_area = cv2.contourArea(largest_yellow)
                if yellow_area > 200:
                    # Verificar circularidad para confirmar que es semáforo
                    perimeter = cv2.arcLength(largest_yellow, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * yellow_area / (perimeter * perimeter)
                        if circularity > 0.3:
                            self.get_logger().info('🟨 Amarillo detectado: REDUCIENDO VELOCIDAD')
                            self.slow_down_pub.publish(Bool(data=True))

            # Imprimir colores detectados
            if colors_detected:
                self.get_logger().info(f'🎯 Colores detectados: {", ".join(colors_detected)}')
            else:
                # Solo mostrar cada 60 frames para no saturar
                if self.frame_count % 60 == 0:
                    self.get_logger().info('❌ No se detectaron colores válidos')

            # Agregar texto de estado en la imagen
            status_text = "ESPERANDO"
            if self.mission_started:
                status_text = "MISION ACTIVA"
            if self.waiting_for_green:
                status_text = "ESPERANDO VERDE"
            
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Frame: {self.frame_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Publicar imagen para debug
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(frame, encoding='bgr8'))

        except Exception as e:
            self.get_logger().error(f'❌ Error en image_callback: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = Semaforo()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 Nodo detenido por el usuario')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

"""import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class Semaforo(Node):
    def __init__(self):
        super().__init__('semaforo_node')
        self.bridge = CvBridge()
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_safe', 10)
        self.debug_pub = self.create_publisher(Image, '/debug_image', 10)
        self.mask_pub = self.create_publisher(Image, '/mask_debug', 10)
        self.subscription = self.create_subscription(Image, '/image_raw', self.image_callback, 10)

        self.mission_started = False
        self.frames_since_green = 0
        self.max_linear_vel = 0.5
        self.max_angular_vel = 1.0

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Rango de colores HSV
        lower_green = np.array([35, 40, 30])   # Verde más oscuro, menos saturado
        upper_green = np.array([90, 255, 255]) # Verde muy brillante, incluso tirando a blanco


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

        kernel = np.ones((5, 5), np.uint8)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)

        self.mask_pub.publish(self.bridge.cv2_to_imgmsg(mask_green, encoding='mono8'))

        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        twist = Twist()
        slow_mode = False

        # 🟡 Amarillo detectado → ir lento
        if contours_yellow:
            largest_yellow = max(contours_yellow, key=cv2.contourArea)
            if cv2.contourArea(largest_yellow) > 500:
                slow_mode = True
                self.get_logger().info('🟡 Amarillo detectado: MODO LENTO ACTIVADO')

        # 🟥 Rojo detectado → detener misión
        if contours_red:
            largest_red = max(contours_red, key=cv2.contourArea)
            if cv2.contourArea(largest_red) > 500:
                self.get_logger().info('🟥 Rojo detectado: DETENIÉNDOSE')
                self.mission_started = False
                self.frames_since_green = 0
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_pub.publish(twist)
                return

        # 🟢 Verde detectado → iniciar misión
        if contours_green:
            largest = max(contours_green, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            # 🟢 Verde detectado → iniciar misión y moverse en círculo
            if area > 800:
                self.get_logger().info(f'🟢 Verde detectado | Área: {area:.2f}')
                
                twist.linear.x = 0.05 if not slow_mode else 0.03
                twist.angular.z = 0.3 if not slow_mode else 0.15  # Giro constante
                
                self.mission_started = True
                self.frames_since_green = 0

                # Dibujar el contorno
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.drawContours(frame, [largest], -1, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        elif self.mission_started:
            # Sigue girando en círculo
            twist.linear.x = 0.05 if not slow_mode else 0.03
            twist.angular.z = 0.3 if not slow_mode else 0.15
            self.get_logger().info('🟢 Verde perdido: SIGUE CÍRCULO...')

        else:
            # Esperando a que aparezca verde
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.cmd_pub.publish(twist)
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(frame, encoding='bgr8'))

def main(args=None):
    rclpy.init(args=args)
    node = Semaforo()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()"""