
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class HttpCameraStreamer(Node):
    def __init__(self):
        super().__init__('http_camera_streamer')
        self.publisher_ = self.create_publisher(Image, '/image_raw', 10)
        self.bridge = CvBridge()
        self.stream_url = 'http://192.168.137.10:5000/car_cam'

        self.cap = cv2.VideoCapture(self.stream_url)
        if not self.cap.isOpened():
            self.get_logger().error(f"❌ No se pudo abrir el stream: {self.stream_url}")
            return
        else:
            self.get_logger().info(f"✅ Conectado al stream: {self.stream_url}")

        self.timer = self.create_timer(1.0 / 30.0, self.publish_frame)

    def publish_frame(self):
        ret, frame = self.cap.read()
        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher_.publish(msg)
        else:
            self.get_logger().warning("⚠️ No se pudo leer frame del stream")

def main(args=None):
    rclpy.init(args=args)
    node = HttpCameraStreamer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()