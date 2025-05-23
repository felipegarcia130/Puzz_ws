import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class YoloPersonTracker(Node):
    def __init__(self):
        super().__init__('yolo_person_tracker_node')
        self.bridge = CvBridge()

        # Sub y Pub
        self.sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.pub = self.create_publisher(Float32MultiArray, '/person_tracking', 10)
        self.debug_pub = self.create_publisher(Image, '/debug_image', 10)

        # Modelo ligero
        self.model = YOLO("yolov8n.pt")

        # Parámetros de optimización
        self.frame_skip = 2
        self.counter = 0

    def image_callback(self, msg):
        self.counter += 1
        if self.counter % self.frame_skip != 0:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Reducción de resolución para procesar más rápido
        frame_resized = cv2.resize(frame, (320, 240))

        results = self.model(frame_resized)[0]
        h, w, _ = frame_resized.shape

        # Filtrar personas
        person_boxes = [box for box in results.boxes if int(box.cls[0]) == 0]
        if not person_boxes:
            return

        # Escoger la persona más ancha
        best = max(person_boxes, key=lambda b: b.xyxy[0][2] - b.xyxy[0][0])
        x1, y1, x2, y2 = best.xyxy[0].tolist()
        cx = (x1 + x2) / 2
        area = (x2 - x1) * (y2 - y1)

        # Publicar datos de seguimiento
        msg_out = Float32MultiArray()
        msg_out.data = [cx - w / 2, area]
        self.pub.publish(msg_out)

        # Dibujar caja en frame reducido
        cv2.rectangle(frame_resized, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        debug_msg = self.bridge.cv2_to_imgmsg(frame_resized, encoding='bgr8')
        self.debug_pub.publish(debug_msg)

def main(args=None):
    rclpy.init(args=args)
    node = YoloPersonTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
