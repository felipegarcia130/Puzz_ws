import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class GreenTracker(Node):
    def __init__(self):
        super().__init__('green_tracker')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.debug_pub = self.create_publisher(Image, '/debug_image', 10)
        self.bridge = CvBridge()

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Rango de color verde
        lower_green = np.array([40, 70, 70])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        twist = Twist()

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area > 500:
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Visual feedback
                    cv2.drawContours(frame, [largest], -1, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                    error_x = cx - frame.shape[1] // 2

                    # 🛑 Criterio para detenerse a ~10 cm (ajustar área según prueba real)
                    if area < 5000:
                        twist.linear.x = 0.08
                        twist.angular.z = -error_x / 300.0
                        self.get_logger().info(f"🟢 Lejos: avanzando rápido | Área: {area:.2f}")
                    elif area < 10000:
                        twist.linear.x = 0.03
                        twist.angular.z = -error_x / 400.0
                        self.get_logger().info(f"🟢 Cerca: avanzando lento | Área: {area:.2f}")
                    else:
                        twist.linear.x = 0.0
                        twist.angular.z = 0.0
                        self.get_logger().info("✅ Muy cerca del verde: deteniéndose")


        self.publisher.publish(twist)

        debug_img = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.debug_pub.publish(debug_img)

def main(args=None):
    rclpy.init(args=args)
    node = GreenTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
