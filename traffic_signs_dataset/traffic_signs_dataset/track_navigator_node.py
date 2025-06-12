#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Int32, Float32, Bool
import math
import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional

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
    box: list
    confidence: float
    approx_dist: Optional[float] = None
    timestamp: Optional[float] = None

# ==================== MENSAJES PERSONALIZADOS (SIMULADOS) ====================

class DetectedSign:
    def __init__(self):
        self.type = 0
        self.box = [0.0, 0.0, 0.0, 0.0]
        self.confidence = 0.0
        self.approx_dist = 0.0
        self.timestamp = 0.0

class DetectedSigns:
    def __init__(self):
        self.header = None
        self.signs = []

class IntersectionData:
    def __init__(self):
        self.header = None
        self.detected = False
        self.center = None
        self.angle = 0.0
        self.confidence = 0
        self.back_detected = False
        self.left_detected = False
        self.right_detected = False
        self.front_detected = False

class StoplightState:
    def __init__(self):
        self.header = None
        self.state = -1
        self.red_detected = False
        self.yellow_detected = False
        self.green_detected = False

class FlagData:
    def __init__(self):
        self.header = None
        self.detected = False
        self.distance = 0.0
        self.end_reached = False

class NavigationCommand:
    def __init__(self):
        self.header = None
        self.state = ""
        self.turn_command = ""
        self.authority = 1.0
        self.stopping = False
        self.throttle = 0.0
        self.yaw = 0.0

# ==================== CLASE TRACK NAVIGATOR ORIGINAL ====================

class TrackNavigator:
    def __init__(self):
        # Estados de la máquina de estados
        self.state = "FOLLOWING"  # FOLLOWING, INTERSECTION_DETECTED, CROSSING, TURNING, RESUMING
        self.intersection_confidence = 0
        self.min_confidence = 5
        
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
        
        # Variables de estado
        self.last_signs = []
        self.last_turn_sign = None
        self.current_light = -1  # -1: desconocido, 0: rojo, 1: amarillo, 2: verde
        self.stopping = False
        self.stop_time = None
        self.stop_duration = 2.0  # segundos detenido
        self.authority = 1.0
        self.last_stoplight = None

    def _apply_sign_speed_control(self):
        """Aplicar control de velocidad basado en señales detectadas"""
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
                    self.authority = 0.5
                elif SignType.YIELD in closest_signs and closest_signs[SignType.YIELD].approx_dist < 0.75:
                    self.authority = 0.5

    def _execute_turn(self):
        """Ejecutar giro según el comando de dirección"""
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

    def navigate(self, signs_data, intersection_data, stoplight_data, flag_data):
        """Máquina de estados para navegación completa con intersecciones"""
        
        # --- DETECCIÓN DE BANDERA ---
        if flag_data and flag_data.detected and flag_data.distance <= 0.40:
            return 0.0, 0.0, True  # Detener completamente
        
        if flag_data and flag_data.end_reached:
            return 0.0, 0.0, True  # Mantener detenido

        # --- DETECCIÓN DE SEMÁFORO ---
        if stoplight_data:
            self.current_light = stoplight_data.state
            if stoplight_data.state == 0:  # ROJO
                self.authority = 0.0
                self.poll = False
            elif stoplight_data.state == 1:  # AMARILLO
                self.authority = 0.5
                self.poll = False
            elif stoplight_data.state == 2:  # VERDE
                self.authority = 1.0
                self.poll = True

        # Verificar si estamos parando por señal STOP
        if self.stopping:
            if self.stop_time is None:
                self.stop_time = time.time()
                return 0.0, 0.0, False
            elif time.time() - self.stop_time < self.stop_duration:
                return 0.0, 0.0, False
            else:
                self.stopping = False
                self.stop_time = None

        # Verificar si se completó el cruce y aplicar acción pendiente
        if self.just_crossed_intersection and self.cross_complete_time is not None:
            if time.time() > self.cross_complete_time:
                self.just_crossed_intersection = False
                if self.pending_action == 'STOP':
                    self.stopping = True
                elif self.pending_action in ['YIELD', 'SLOW']:
                    self.authority = 0.5
                self.pending_action = None

        # Procesar señales detectadas
        if signs_data:
            self.last_signs = []
            for sign_msg in signs_data.signs:
                sign = Sign(
                    type=SignType(sign_msg.type),
                    box=sign_msg.box,
                    confidence=sign_msg.confidence,
                    approx_dist=sign_msg.approx_dist if sign_msg.approx_dist > 0 else None,
                    timestamp=sign_msg.timestamp if sign_msg.timestamp > 0 else None
                )
                self.last_signs.append(sign)
            
            # Actualizar comando de giro si hay señales direccionales
            turn_signs = [s for s in self.last_signs if 0 <= s.type.value <= 3]
            if turn_signs:
                # Tomar la señal con mayor confianza
                best_turn_sign = max(turn_signs, key=lambda s: s.confidence)
                self.turn_command = best_turn_sign.type
                self.last_turn_sign = best_turn_sign

        # Detectar intersecciones
        intersection = None
        if intersection_data:
            intersection = intersection_data
            
            # Sistema de confianza
            if intersection_data.detected:
                self.intersection_confidence = min(self.intersection_confidence + 1, self.min_confidence + 2)
            else:
                self.intersection_confidence = max(self.intersection_confidence - 1, 0)

        # ============ MÁQUINA DE ESTADOS ============
        
        if self.state == "FOLLOWING":
            if self.current_light == 0:  # ROJO
                return 0.0, 0.0, False  # 🚦 Detener completamente
            elif self.current_light == 1:  # AMARILLO
                self.authority = 0.4  # ⚠️ Reducir velocidad
            
            # Estado normal: seguir líneas
            self.authority = 1.0
            
            # Controlar velocidad basado en señales
            self._apply_sign_speed_control()
            
            # Verificar si hay intersección detectada
            if self.intersection_confidence >= self.min_confidence and intersection:
                if self.turn_command:  # Solo si tenemos una señal de dirección
                    self.state = "INTERSECTION_DETECTED"
            
            # Señalar que debe seguir línea normalmente
            return None, None, False  # None significa usar line follower
            
        elif self.state == "INTERSECTION_DETECTED":
            # Detectamos intersección, prepararse para cruzar
            self.authority = 0.8  # Reducir velocidad
            
            # Simular alineación con intersección (en el código original usa PID)
            # Aquí simplificamos: cuando esté cerca, empezar a cruzar
            if self.current_light == 2 or self.current_light == -1:  # Verde O sin semáforo
                self.state = "CROSSING"
                self.crossing_timer = self.crossing_duration
            elif self.current_light == 1:  # Amarillo - lógica realista
                # Si está MUY cerca, cruza; si no, se detiene
                self.state = "CROSSING"
                self.crossing_timer = self.crossing_duration
            elif self.current_light == 0:  # Rojo
                return 0.0, 0.0, False  # Mantenerse detenido
                
            return 0.15, 0.0, False  # Velocidad constante hacia intersección
                
        elif self.state == "CROSSING":
            # Cruzar la intersección en línea recta
            throttle = 0.15  # Velocidad constante para cruzar
            yaw = 0.0       # Sin giro, ir derecho
            
            self.crossing_timer -= 1
            if self.crossing_timer <= 0:
                self.state = "TURNING"
                self.turning_timer = self.turning_duration
            
            return throttle, yaw, False
                
        elif self.state == "TURNING":
            # Ejecutar el giro según la señal
            throttle, yaw = self._execute_turn()
            
            self.turning_timer -= 1
            if self.turning_timer <= 0:
                self.state = "RESUMING"
                self.resuming_timer = self.resuming_duration
                self.just_crossed_intersection = True
                self.cross_complete_time = time.time() + 1.5  # 1.5s después del giro
            
            return throttle, yaw, False
                
        elif self.state == "RESUMING":
            # Estabilizar y volver a seguimiento normal
            self.resuming_timer -= 1
            if self.resuming_timer <= 0:
                self.state = "FOLLOWING"
                self.turn_command = None  # Limpiar comando
                self.intersection_confidence = 0  # Reset confianza
            
            # Señalar que debe seguir línea normalmente
            return None, None, False  # None significa usar line follower
        
        # Default: seguir línea
        return None, None, False

# ==================== NODO NAVEGADOR PRINCIPAL ====================

class TrackNavigatorNode(Node):
    def __init__(self):
        super().__init__('track_navigator_node')
        
        # Parámetros
        self.declare_parameter('min_confidence', 3)
        self.declare_parameter('crossing_duration', 28)
        self.declare_parameter('turning_duration', 30)
        self.declare_parameter('resuming_duration', 15)
        
        min_confidence = self.get_parameter('min_confidence').get_parameter_value().integer_value
        crossing_duration = self.get_parameter('crossing_duration').get_parameter_value().integer_value
        turning_duration = self.get_parameter('turning_duration').get_parameter_value().integer_value
        resuming_duration = self.get_parameter('resuming_duration').get_parameter_value().integer_value
        
        # Inicializar navegador
        self.navigator = TrackNavigator()
        self.navigator.min_confidence = min_confidence
        self.navigator.crossing_duration = crossing_duration
        self.navigator.turning_duration = turning_duration
        self.navigator.resuming_duration = resuming_duration
        
        # Variables para almacenar datos de sensores
        self.latest_signs = None
        self.latest_intersection = None
        self.latest_stoplight = None
        self.latest_flag = None
        self.latest_line_cmd = None
        
        # ROS2 setup
        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_safe', 10)
        self.nav_cmd_pub = self.create_publisher(NavigationCommand, '/navigation_command', 10)
        self.status_pub = self.create_publisher(String, '/robot_status', 10)
        
        # Subscribers
        self.signs_sub = self.create_subscription(
            DetectedSigns, '/detected_signs', self.signs_callback, 10)
        self.intersection_sub = self.create_subscription(
            IntersectionData, '/intersection_data', self.intersection_callback, 10)
        self.stoplight_sub = self.create_subscription(
            StoplightState, '/stoplight_state', self.stoplight_callback, 10)
        self.flag_sub = self.create_subscription(
            FlagData, '/flag_data', self.flag_callback, 10)
        self.line_cmd_sub = self.create_subscription(
            Twist, '/line_following_cmd', self.line_cmd_callback, 10)
        
        # Timer para navegación principal
        self.create_timer(0.033, self.navigation_timer_callback)  # ~30Hz
        
        self.get_logger().info('Track Navigator Node initialized')
    
    def signs_callback(self, msg):
        self.latest_signs = msg
    
    def intersection_callback(self, msg):
        self.latest_intersection = msg
    
    def stoplight_callback(self, msg):
        self.latest_stoplight = msg
    
    def flag_callback(self, msg):
        self.latest_flag = msg
    
    def line_cmd_callback(self, msg):
        self.latest_line_cmd = msg
    
    def navigation_timer_callback(self):
        try:
            # Ejecutar navegación
            throttle, yaw, stop = self.navigator.navigate(
                self.latest_signs,
                self.latest_intersection,
                self.latest_stoplight,
                self.latest_flag
            )
            
            # Crear comando final
            final_cmd = Twist()
            
            if stop:
                # Parar completamente
                final_cmd.linear.x = 0.0
                final_cmd.angular.z = 0.0
            elif throttle is None and yaw is None:
                # Usar comando de line follower con authority
                if self.latest_line_cmd:
                    final_cmd.linear.x = float(self.latest_line_cmd.linear.x * self.navigator.authority)
                    final_cmd.angular.z = float(self.latest_line_cmd.angular.z * self.navigator.authority)
            else:
                # Usar comando de navegación
                final_cmd.linear.x = float(throttle)
                final_cmd.angular.z = float(yaw)
            
            # Publicar comando final
            self.cmd_pub.publish(final_cmd)
            
            # Crear y publicar comando de navegación para otros nodos
            nav_cmd = NavigationCommand()
            nav_cmd.state = self.navigator.state
            nav_cmd.turn_command = self.navigator.turn_command.name if self.navigator.turn_command else "NONE"
            nav_cmd.authority = self.navigator.authority
            nav_cmd.stopping = self.navigator.stopping
            nav_cmd.throttle = final_cmd.linear.x
            nav_cmd.yaw = final_cmd.angular.z
            self.nav_cmd_pub.publish(nav_cmd)
            
            # Publicar estado
            status_msg = String()
            turn_cmd = self.navigator.turn_command.name if self.navigator.turn_command else "NONE"
            signs_count = len(self.navigator.last_signs) if self.navigator.last_signs else 0
            status_msg.data = f"STATE:{self.navigator.state}, TURN:{turn_cmd}, v={final_cmd.linear.x:.3f}, w={math.degrees(final_cmd.angular.z):.1f}°/s, signs={signs_count}"
            self.status_pub.publish(status_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error in navigation: {str(e)}')
            # Comando de seguridad
            stop_cmd = Twist()
            self.cmd_pub.publish(stop_cmd)

def main(args=None):
    rclpy.init(args=args)
    node = TrackNavigatorNode()
    
    try:
        node.get_logger().info("🚀 Track Navigator Node started")
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Stopping Track Navigator Node...")
    finally:
        # Detener el robot antes de cerrar
        stop_cmd = Twist()
        node.cmd_pub.publish(stop_cmd)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()