import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
import yaml
import os
import math
import tf_transformations

class PathGenerator(Node):
    def __init__(self):
        super().__init__('path_generator')

        self.pub = self.create_publisher(PoseStamped, 'goal_pose', 10)
        self.sub = self.create_subscription(Bool, 'completed_point', self.confirm_callback, 10)

        self.declare_parameter('path_file', '')
        file_path = self.get_parameter('path_file').get_parameter_value().string_value

        if not file_path or not os.path.isfile(file_path):
            self.get_logger().error('No se especificó un archivo de trayectoria válido')
            return

        with open(file_path, 'r') as f:
            self.points = yaml.safe_load(f)['points']

        self.index = 0
        self.ready = True  # Para publicar el primer punto de inmediato

        self.get_logger().info('🚀 Generador de trayectoria listo.')

        # Timer para checar si debe publicar (si no usamos temporizador, no arranca el primero)
        self.timer = self.create_timer(0.5, self.try_publish_next)

    def confirm_callback(self, msg: Bool):
        if msg.data:
            self.ready = True  # Se recibió confirmación, listo para siguiente

    def try_publish_next(self):
        if self.ready and self.index < len(self.points):
            point = self.points[self.index]
            msg = PoseStamped()
            msg.header.frame_id = 'map'
            msg.header.stamp = self.get_clock().now().to_msg()

            msg.pose.position.x = float(point['x'])
            msg.pose.position.y = float(point['y'])

            # Orientación hacia siguiente punto (si hay uno más)
            if self.index + 1 < len(self.points):
                next_pt = self.points[self.index + 1]
                dx = next_pt['x'] - point['x']
                dy = next_pt['y'] - point['y']
                yaw = math.atan2(dy, dx)
            else:
                yaw = 0.0

            q = tf_transformations.quaternion_from_euler(0, 0, yaw)
            msg.pose.orientation.x = q[0]
            msg.pose.orientation.y = q[1]
            msg.pose.orientation.z = q[2]
            msg.pose.orientation.w = q[3]

            self.pub.publish(msg)
            self.get_logger().info(f'Publicado punto {self.index + 1}/{len(self.points)}')
            self.index += 1
            self.ready = False

        elif self.index >= len(self.points):
            self.get_logger().info('Trayectoria completa.')
            self.timer.cancel()

def main(args=None):
    rclpy.init(args=args)
    node = PathGenerator()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
