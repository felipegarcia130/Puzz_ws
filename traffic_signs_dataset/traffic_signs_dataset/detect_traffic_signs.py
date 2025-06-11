"""from ultralytics import YOLO
import cv2
import time
from datetime import datetime
import os

# Carga el modelo entrenado
model = YOLO("traffic_signs_dataset/best.pt")

# Abre la cámara (0 = webcam por defecto)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: no se pudo abrir la cámara.")
    exit()

# Obtener propiedades del video para la grabación
fps_camera = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # FPS de la cámara, default 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"📹 Resolución: {width}x{height} a {fps_camera} FPS")

# Crear carpeta para videos si no existe
output_dir = "videos_deteccion"
os.makedirs(output_dir, exist_ok=True)

# Configurar grabación de video
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"{output_dir}/deteccion_{timestamp}.mp4"

# Codec y configuración del video de salida
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # También puedes usar 'XVID'
out = cv2.VideoWriter(output_filename, fourcc, fps_camera, (width, height))

if not out.isOpened():
    print("❌ Error: no se pudo crear el archivo de video.")
    cap.release()
    exit()

# Crear ventana una sola vez fuera del bucle
window_name = "Detección en tiempo real"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

print("🎥 Iniciando detección... Presiona 'q' para salir")
print(f"📼 Grabando video en: {output_filename}")

# Variables para el conteo de frames
frame_count = 0
start_total_time = time.time()

# Bucle de detección en tiempo real
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: no se pudo leer el frame de la cámara")
            break
                
        start_time = time.time()
                
        # Realiza la predicción
        results = model.predict(source=frame, conf=0.5, verbose=False, fliplr=False)
                
        # Muestra los resultados con bounding boxes
        annotated_frame = results[0].plot()
                
        # Muestra FPS en pantalla
        fps = 1 / (time.time() - start_time)
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Agregar indicador de grabación
        cv2.putText(annotated_frame, "🔴 GRABANDO", (width - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Agregar contador de frames grabados
        frame_count += 1
        cv2.putText(annotated_frame, f"Frames: {frame_count}", (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
        # Guardar frame en el video
        out.write(annotated_frame)
                
        # Visualización usando la ventana creada
        cv2.imshow(window_name, annotated_frame)
                
        # Presiona 'q' para salir
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # 'q' o ESC
            break
                
        # Verificar si la ventana fue cerrada
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

except KeyboardInterrupt:
    print("\n🛑 Detenido por el usuario")
except Exception as e:
    print(f"❌ Error durante la ejecución: {e}")
finally:
    # Calcular estadísticas finales
    total_time = time.time() - start_total_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    # Libera recursos
    print("🧹 Liberando recursos...")
    cap.release()
    out.release()  # ¡Importante! Liberar el writer del video
    cv2.destroyAllWindows()
    
    # Mostrar estadísticas
    print("✅ Programa terminado correctamente")
    print(f"📊 Estadísticas de grabación:")
    print(f"   • Frames grabados: {frame_count}")
    print(f"   • Duración: {total_time:.1f} segundos")
    print(f"   • FPS promedio: {avg_fps:.1f}")
    print(f"   • Video guardado en: {output_filename}")
    
    # Verificar si el archivo se creó correctamente
    if os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
        size_mb = os.path.getsize(output_filename) / (1024 * 1024)
        print(f"   • Tamaño del archivo: {size_mb:.2f} MB")
    else:
        print("⚠️  Advertencia: El archivo de video puede no haberse guardado correctamente")"""

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import String, Header
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Mensaje personalizado para detecciones (puedes crear un paquete de mensajes separado)
from std_msgs.msg import Float64MultiArray

class TrafficSignDetector(Node):
    def __init__(self):
        super().__init__('traffic_sign_detector')
        
        # Parámetros configurables para PuzzleBot
        self.declare_parameter('model_path', 'traffic_signs_dataset/best.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('camera_topic', '/image_raw')  # Tópico del PuzzleBot
        self.declare_parameter('output_topic', '/debug_image')  # Tópico de salida del PuzzleBot
        self.declare_parameter('detections_topic', '/traffic_signs/detections')
        
        # Obtener parámetros
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.detections_topic = self.get_parameter('detections_topic').get_parameter_value().string_value
        
        # Inicializar el modelo YOLO
        try:
            self.model = YOLO(model_path)
            self.get_logger().info(f'✅ Modelo YOLO cargado desde: {model_path}')
        except Exception as e:
            self.get_logger().error(f'❌ Error al cargar el modelo: {e}')
            return
        
        # Inicializar CV Bridge
        self.bridge = CvBridge()
        
        # Publishers
        self.image_pub = self.create_publisher(Image, self.output_topic, 10)  # /debug_image
        self.detections_pub = self.create_publisher(String, self.detections_topic, 10)
        
        # Subscriber para imágenes desde el PuzzleBot
        self.image_sub = self.create_subscription(Image, self.camera_topic, self.image_callback, 10)  # /image_raw
        
        # Variables para FPS y estadísticas
        self.last_time = time.time()
        self.frame_count = 0
        self.total_detections = 0
        
        # Status del stream
        self.stream_active = False
        self.last_frame_time = time.time()
        
        # Timer para verificar status del stream
        self.status_timer = self.create_timer(2.0, self.check_stream_status)
        
        self.get_logger().info(f'🎥 Detector de señales iniciado para PuzzleBot')
        self.get_logger().info(f'📡 Suscrito a: {self.camera_topic}')
        self.get_logger().info(f'📤 Publicando en: {self.output_topic}')
        
    def check_stream_status(self):
        #Verifica si el stream del PuzzleBot está activo
        current_time = time.time()
        time_since_last_frame = current_time - self.last_frame_time
        
        if time_since_last_frame > 5.0:  # Sin frames por 5 segundos
            if self.stream_active:
                self.get_logger().warn(f'⚠️ Stream inactivo desde hace {time_since_last_frame:.1f}s')
                self.get_logger().warn(f'🔍 Verificar que el PuzzleBot esté publicando en: {self.camera_topic}')
                self.stream_active = False
        else:
            if not self.stream_active:
                self.get_logger().info('✅ Stream del PuzzleBot reactivado')
                self.stream_active = True
    
    def image_callback(self, msg):
        #Callback para procesar imágenes desde el stream del PuzzleBot
        try:
            # Actualizar timestamp del último frame
            self.last_frame_time = time.time()
            
            # Convertir mensaje ROS a OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # Log de primera conexión
            if not self.stream_active:
                height, width = frame.shape[:2]
                self.get_logger().info(f'📹 Stream conectado - Resolución: {width}x{height}')
                self.stream_active = True
            
            self.process_frame(frame, msg.header)
            
        except Exception as e:
            self.get_logger().error(f'Error al procesar imagen del PuzzleBot: {e}')
    
    def process_frame(self, frame, header=None):
        #Procesa un frame con YOLO y publica los resultados
        start_time = time.time()
        
        try:
            # Realizar predicción
            results = self.model.predict(
                source=frame, 
                conf=self.confidence_threshold, 
                verbose=False, 
                fliplr=False
            )
            
            # Anotar frame con detecciones
            annotated_frame = results[0].plot()
            
            # Calcular y mostrar FPS
            fps = 1 / (time.time() - start_time)
            
            # Agregar información del PuzzleBot al frame
            cv2.putText(annotated_frame, f"PuzzleBot FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Agregar timestamp
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            cv2.putText(annotated_frame, f"Time: {timestamp}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Agregar contador de detecciones
            num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
            self.total_detections += num_detections
            cv2.putText(annotated_frame, f"Detections: {num_detections}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Publicar imagen anotada
            self.publish_image(annotated_frame, header)
            
            # Publicar detecciones como texto
            self.publish_detections(results[0], header)
            
            # Log de información cada 30 frames
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                self.get_logger().info(f'📊 FPS: {fps:.1f}, Detecciones: {num_detections}, Total: {self.total_detections}')
                
        except Exception as e:
            self.get_logger().error(f'Error en procesamiento: {e}')
    
    def publish_image(self, frame, original_header=None):
        #Publica la imagen anotada como mensaje ROS
        try:
            # Convertir OpenCV a mensaje ROS
            msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
            
            # Usar header original si está disponible, sino crear uno nuevo
            if original_header:
                msg.header = original_header
            else:
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = 'camera_frame'
            
            self.image_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f'Error al publicar imagen: {e}')
    
    def publish_detections(self, result, original_header=None):
        #Publica las detecciones como mensaje de texto
        try:
            detections_info = {
                'timestamp': time.time(),
                'robot': 'puzzlebot',
                'detections': []
            }
            
            if result.boxes is not None:
                boxes = result.boxes
                for i, box in enumerate(boxes):
                    # Obtener información de la detección
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self.model.names[cls] if cls < len(self.model.names) else f"class_{cls}"
                    
                    # Coordenadas del bounding box
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    
                    detection = {
                        'id': i,
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2],
                        'center': [(x1+x2)/2, (y1+y2)/2],
                        'area': (x2-x1) * (y2-y1)
                    }
                    detections_info['detections'].append(detection)
            
            # Publicar como String JSON
            msg = String()
            msg.data = str(detections_info)
            self.detections_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f'Error al publicar detecciones: {e}')
    
    def destroy_node(self):
        #Limpieza al destruir el nodo
        self.get_logger().info('🧹 Liberando recursos del detector PuzzleBot...')
        cv2.destroyAllWindows()
        super().destroy_node()
        self.get_logger().info('✅ Nodo PuzzleBot terminado correctamente')


def main(args=None):
    rclpy.init(args=args)
    
    try:
        detector = TrafficSignDetector()
        
        # Verificar si el nodo se inicializó correctamente
        if hasattr(detector, 'model'):
            rclpy.spin(detector)
        else:
            detector.get_logger().error('❌ Fallo en la inicialización del nodo')
            
    except KeyboardInterrupt:
        print('\n🛑 Detenido por el usuario')
    except Exception as e:
        print(f'❌ Error durante la ejecución: {e}')
    finally:
        if 'detector' in locals():
            detector.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
