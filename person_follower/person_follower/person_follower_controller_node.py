import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from simple_pid import PID

class PersonFollower(Node):
    def __init__(self):
        super().__init__('person_follower_controller_node')

        # Suscripción a datos de tracking
        self.sub = self.create_subscription(Float32MultiArray, '/person_tracking', self.tracking_callback, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel_safe', 10)

        # PID para orientación (giro)
        self.pid_yaw = PID(0.005, 0, 0.001, setpoint=0)
        self.pid_yaw.output_limits = (-0.5, 0.5)

        # PID para distancia (avance)
        self.target_area = 8000
        self.pid_thr = PID(0.00005, 0, 0.00001, setpoint=self.target_area)
        self.pid_thr.output_limits = (0.0, 0.2)

        # Timeout si se pierde la persona
        self.last_detection_time = self.get_clock().now()
        self.timer = self.create_timer(0.1, self.check_timeout)

        # Umbral para definir si estamos bien alineados (± px)
        self.yaw_alignment_threshold = 30  # para imagen de 320 px de ancho

    def tracking_callback(self, msg):
        offset, area = msg.data
        yaw_error = offset
        distance_error = area

        twist = Twist()

        # 🧠 Primero alinear, luego avanzar si ya está centrado
        if abs(yaw_error) > self.yaw_alignment_threshold:
            # Solo girar
            twist.angular.z = -self.pid_yaw(yaw_error)
            twist.linear.x = 0.0
        else:
            # Ya está alineado, puede avanzar
            twist.angular.z = -self.pid_yaw(yaw_error)
            twist.linear.x = self.pid_thr(distance_error)

        self.pub.publish(twist)
        self.last_detection_time = self.get_clock().now()

    def check_timeout(self):
        now = self.get_clock().now()
        elapsed = (now - self.last_detection_time).nanoseconds * 1e-9
        if elapsed > 1.0:
            self.get_logger().warn("Persona perdida, deteniendo robot.")
            self.pub.publish(Twist())

def main(args=None):
    rclpy.init(args=args)
    node = PersonFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
