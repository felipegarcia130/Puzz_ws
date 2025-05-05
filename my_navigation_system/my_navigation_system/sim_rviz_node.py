import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
import tf_transformations
import math
import time

class RVizSimulator(Node):
    def __init__(self):
        super().__init__('rviz_simulator_node')
        self.create_subscription(PoseStamped, 'goal_pose', self.goal_callback, 10)
        self.marker_pub = self.create_publisher(Marker, 'visualization_marker', 10)

        self.current_pos = [0.0, 0.0]
        self.target_pos = None
        self.start_time = None
        self.duration = 2.0  # segundos para llegar al siguiente punto

        self.timer = self.create_timer(0.05, self.update)

    def goal_callback(self, msg):
        self.target_pos = [msg.pose.position.x, msg.pose.position.y]
        self.start_time = time.time()
        self.get_logger().info(f'📍 Nuevo destino RViz: {self.target_pos}')

    def update(self):
        if self.target_pos is None:
            return

        t = time.time() - self.start_time
        alpha = min(t / self.duration, 1.0)  # factor de interpolación entre 0 y 1

        # Interpolación lineal
        x = self.current_pos[0] + (self.target_pos[0] - self.current_pos[0]) * alpha
        y = self.current_pos[1] + (self.target_pos[1] - self.current_pos[1]) * alpha

        if alpha >= 1.0:
            self.current_pos = self.target_pos
            self.target_pos = None

        # Publicar marcador
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "sim_rviz"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.1
        q = tf_transformations.quaternion_from_euler(0, 0, 0)
        marker.pose.orientation.x = q[0]
        marker.pose.orientation.y = q[1]
        marker.pose.orientation.z = q[2]
        marker.pose.orientation.w = q[3]

        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = RVizSimulator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
