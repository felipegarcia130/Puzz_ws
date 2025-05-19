# Author: AECL
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from cv_bridge import CvBridge

import cv2
import numpy as np
import grpc
import google.protobuf.empty_pb2
from . import te3002b_pb2
from . import te3002b_pb2_grpc


class RESTImageNode(Node):
    def __init__(self):
        super().__init__('rpc_image_node')

        # Inicializar gRPC
        self._addr = "localhost"
        self.channel = grpc.insecure_channel(f"{self._addr}:7072")
        self.stub = te3002b_pb2_grpc.TE3002BSimStub(self.channel)

        # QoS para /cmd_vel
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publicador de imagen
        #self.publisher_ = self.create_publisher(Image, 'Image', 10)
        self.publisher_ = self.create_publisher(Image, '/image_raw', 10)

        self.bridge = CvBridge()

        # Subscripción a /cmd_vel
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.listener_callback,
            qos_profile=qos_profile
        )

        # Timer para publicar imágenes y comandos
        self.timer = self.create_timer(0.1, self.timer_callback)

        # Estados internos
        self.twist = [0, 0, 0, 0, 0, 0]
        self.cv_image = None
        self.datacmd = te3002b_pb2.CommandData()

        # Verificación de GUI disponible
        self.gui_enabled = self._check_gui_support()

    def _check_gui_support(self):
        """Verifica si OpenCV soporta imshow."""
        try:
            cv2.namedWindow("Test")
            cv2.destroyWindow("Test")
            return True
        except cv2.error as e:
            self.get_logger().warn("OpenCV GUI not supported. Running in headless mode.")
            return False

    def listener_callback(self, msg):
        self.twist = [
            msg.linear.x, msg.linear.y, msg.linear.z,
            msg.angular.x, msg.angular.y, msg.angular.z
        ]
        self.get_logger().debug(f"Received Twist: {self.twist}")

    def timer_callback(self):
        # Obtener imagen del simulador
        try:
            req = google.protobuf.empty_pb2.Empty()
            result = self.stub.GetImageFrame(req)
            img_buffer = np.frombuffer(result.data, np.uint8)
            img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
            self.cv_image = img

            if self.cv_image is not None:
                # Publicar imagen
                ros_image = self.bridge.cv2_to_imgmsg(self.cv_image, "bgr8")
                self.publisher_.publish(ros_image)
                self.get_logger().info('Published image')

                # Mostrar ventana si se puede
                if self.gui_enabled:
                    cv2.imshow("Robot Camera View", self.cv_image)
                    cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error retrieving or decoding image: {e}")
            return

        # Enviar comando al simulador
        try:
            self.datacmd.linear.x = self.twist[2]
            self.datacmd.linear.y = self.twist[1]
            self.datacmd.linear.z = self.twist[0]
            self.datacmd.angular.x = self.twist[3]
            self.datacmd.angular.y = self.twist[4]
            self.datacmd.angular.z = -self.twist[5] * 2

            self.stub.SetCommand(self.datacmd)
            self.get_logger().debug("Sent command to simulator")
        except Exception as e:
            self.get_logger().error(f"Error sending command: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = RESTImageNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node by user interruption.")
    finally:
        node.destroy_node()
        try:
            node.channel.close()
        except Exception:
            pass
        if node.gui_enabled:
            cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
