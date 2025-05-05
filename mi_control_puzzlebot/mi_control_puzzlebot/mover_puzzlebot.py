import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time
import math
import sys
import select
import termios
import tty

class MoverPuzzlebot(Node):
    def __init__(self):
        super().__init__('mover_puzzlebot')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.mover_ruta)

        # Ruta: avanzar y girar (cuadrado)
        self.ruta_base = [
            {'x': 0.2, 'z': 0.0, 'duracion': 2},
            {'x': 0.0, 'z': math.pi/4, 'duracion': 2.15},
        ]

        self.veces_repetir = 1
        self.etapas = self.ruta_base * 4 * self.veces_repetir

        self.index = 0
        self.start_time = time.time()
        self.frenando = False

    def mover_ruta(self):
        if self._tecla_presionada() == 'k':
            self.get_logger().info('🛑 Tecla "k" detectada. Deteniendo el robot.')
            self.publisher_.publish(Twist())
            self.timer.cancel()
            return

        if self.index >= len(self.etapas):
            self.publisher_.publish(Twist())
            self.get_logger().info('✅ Ruta cuadrada repetida 5 veces. Terminada.')
            return

        tiempo_actual = time.time()
        etapa = self.etapas[self.index]

        if self.frenando:
            if tiempo_actual - self.start_time >= 0.5:
                self.index += 1
                self.start_time = tiempo_actual
                self.frenando = False
            else:
                self.publisher_.publish(Twist())
            return

        if tiempo_actual - self.start_time > etapa['duracion']:
            self.publisher_.publish(Twist())
            self.start_time = tiempo_actual
            self.frenando = True
            return

        twist = Twist()
        twist.linear.x = etapa['x']
        twist.angular.z = etapa['z']
        self.publisher_.publish(twist)

    def _tecla_presionada(self):
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if dr:
            return sys.stdin.read(1)
        return None


def main(args=None):
    rclpy.init(args=args)
    nodo = MoverPuzzlebot()

    # Configura terminal para lectura inmediata (sin Enter)
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    rclpy.spin(nodo)

    # Restaurar la terminal al final
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    nodo.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
