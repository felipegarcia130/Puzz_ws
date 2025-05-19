import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import numpy as np
import cv2

class GStreamerBridgeNode(Node):
    def __init__(self):
        super().__init__('gstreamer_bridge_node')
        self.publisher = self.create_publisher(Image, '/image_raw', 10)
        self.bridge = CvBridge()

        Gst.init(None)

        self.pipeline = Gst.parse_launch(
            'udpsrc port=5001 caps="application/x-rtp, media=video, encoding-name=H264, payload=96" ! '
            'rtpjitterbuffer ! rtph264depay ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink'
        )


        self.appsink = self.pipeline.get_by_name("sink")
        self.appsink.set_property("emit-signals", True)
        self.appsink.connect("new-sample", self.on_new_sample)

        self.pipeline.set_state(Gst.State.PLAYING)
        self.get_logger().info("🎥 GStreamer pipeline iniciada en UDP:5001")

    def on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        caps = sample.get_caps()
        arr = self.gst_buffer_to_ndarray(buf, caps)
        if arr is not None:
            msg = self.bridge.cv2_to_imgmsg(arr, encoding='bgr8')
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera'
            self.publisher.publish(msg)
        return Gst.FlowReturn.OK

    def gst_buffer_to_ndarray(self, buf, caps):
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return None
        try:
            width = caps.get_structure(0).get_value('width')
            height = caps.get_structure(0).get_value('height')
            data = np.frombuffer(map_info.data, np.uint8)
            frame = data.reshape((height, width, 3))
            return frame.copy()
        finally:
            buf.unmap(map_info)

    def destroy_node(self):
        self.pipeline.set_state(Gst.State.NULL)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = GStreamerBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
