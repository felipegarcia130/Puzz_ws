import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Bool  # Mensaje para notificar punto completado
import math
import time

class Controller(Node):
    def __init__(self):
        super().__init__('controller')

        # Estados
        self.ESPERANDO = 0
        self.GIRANDO = 1
        self.AVANZANDO = 2

        self.estado = self.ESPERANDO
        self.queue = []

        self.sub = self.create_subscription(PoseStamped, 'goal_pose', self.goal_callback, 10)
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.done_pub = self.create_publisher(Bool, 'completed_point', 10)

        self.timer = self.create_timer(0.1, self.control_loop)

        self.last_x = 0.0
        self.last_y = 0.0
        self.theta = 0.0

        self.target = None
        self.t_fin = None
        self.vel_actual = Twist()

        self.get_logger().info('✅ Controlador con máquina de estados y cola listo.')

    def goal_callback(self, msg):
        self.queue.append(msg.pose)
        self.get_logger().info(f'📥 Punto agregado a la cola ({len(self.queue)} en espera)')

    def control_loop(self):
        now = time.time()

        if self.estado == self.ESPERANDO and self.queue:
            self.target = self.queue.pop(0)

            dx = self.target.position.x - self.last_x
            dy = self.target.position.y - self.last_y

            self.dist = math.hypot(dx, dy)
            self.ang_obj = math.atan2(dy, dx)
            self.ang_rot = math.atan2(math.sin(self.ang_obj - self.theta), math.cos(self.ang_obj - self.theta))

            if abs(self.ang_rot) > 0.01:
                self.t_rot = abs(self.ang_rot) / 0.6
                self.estado = self.GIRANDO
                self.t_fin = now + self.t_rot
                self.vel_actual = Twist()
                self.vel_actual.angular.z = math.copysign(0.6, self.ang_rot)
                self.get_logger().info(f'🌀 Estado 1: Girando {self.ang_rot:.2f} rad por {self.t_rot:.2f} s')
            else:
                self.estado = self.AVANZANDO
                self.t_fin = now + (self.dist / 0.15)
                self.vel_actual = Twist()
                self.vel_actual.linear.x = 0.15
                self.get_logger().info(f'🚗 Estado 2: Avanzando directo {self.dist:.2f} m')

        elif self.estado == self.GIRANDO:
            if now >= self.t_fin:
                self.theta += self.ang_rot
                self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))  # normalizar
                self.estado = self.AVANZANDO
                self.t_fin = now + (self.dist / 0.15)
                self.vel_actual = Twist()
                self.vel_actual.linear.x = 0.15
                self.get_logger().info(f'🚗 Estado 2: Ahora avanzando {self.dist:.2f} m')

        elif self.estado == self.AVANZANDO:
            if now >= self.t_fin:
                self.vel_actual = Twist()
                self.last_x = self.target.position.x
                self.last_y = self.target.position.y
                self.estado = self.ESPERANDO
                self.get_logger().info('✅ Estado 0: Punto alcanzado.')

                # Publicar punto completado
                msg = Bool()
                msg.data = True
                self.done_pub.publish(msg)

        # Publicar comando actual
        self.pub.publish(self.vel_actual)

def main(args=None):
    rclpy.init(args=args)
    node = Controller()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
