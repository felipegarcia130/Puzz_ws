#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import TwistStamped, PoseStamped
from rclpy.qos import qos_profile_sensor_data
import math

class PoseEstimator(Node):
    def __init__(self):
        super().__init__('pose_estimator')

        # → Declare the new parameter
        self.declare_parameter('angular_factor', 0.9636593357174979)
        self.angular_factor = self.get_parameter('angular_factor').value

        # → React to parameter changes
        self.add_on_set_parameters_callback(self._on_param_update)

        # Velocity
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

        # Pose (internal)
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Time tracking
        self.last_update_time = self.get_clock().now()

        # Subscriber
        self.subscription = self.create_subscription(
            TwistStamped,
            '/robot_vel',
            self.robot_vel_callback,
            qos_profile_sensor_data
        )

        # Pose publisher
        self.pose_publisher = self.create_publisher(
            PoseStamped,
            '/estimated_pose',
            10
        )

        # Timers
        self.sampling_timer = self.create_timer(0.0025, self.sampling_loop)  # 400 Hz
        self.print_timer    = self.create_timer(0.2,    self.print_loop)     #   5 Hz

        self.get_logger().info('Pose Estimator node started and subscribed to /robot_vel')

    def _on_param_update(self, params):
        for p in params:
            if p.name == 'angular_factor':
                self.angular_factor = p.value
        return SetParametersResult(successful=True)

    def robot_vel_callback(self, msg: TwistStamped):
        self.linear_velocity  = msg.twist.linear.x
        self.angular_velocity = msg.twist.angular.z

    def sampling_loop(self):
        now = self.get_clock().now()
        dt  = (now - self.last_update_time).nanoseconds * 1e-9

        # Integrate internal theta
        self.theta += self.angular_velocity * dt

        # Integrate x,y
        self.x += self.linear_velocity * math.cos(self.theta) * dt
        self.y += self.linear_velocity * math.sin(self.theta) * dt

        self.last_update_time = now

        # Publish scaled pose
        scaled_theta = self.theta * self.angular_factor

        pose_msg = PoseStamped()
        pose_msg.header.stamp    = now.to_msg()
        pose_msg.header.frame_id = 'odom'

        pose_msg.pose.position.x = self.x
        pose_msg.pose.position.y = self.y
        pose_msg.pose.position.z = 0.0

        # quaternion from scaled_theta
        pose_msg.pose.orientation.z = math.sin(scaled_theta / 2.0)
        pose_msg.pose.orientation.w = math.cos(scaled_theta / 2.0)

        self.pose_publisher.publish(pose_msg)

    def print_loop(self):
        # scaled theta in degrees
        theta_deg = (self.theta * self.angular_factor) * 180.0 / math.pi

        print(
            f"Linear vel: {self.linear_velocity:.2f} m/s | "
            f"Angular vel: {self.angular_velocity * 180/math.pi:.2f}°/s | "
            f"Theta: {theta_deg:.2f}° | "
            f"X: {self.x:.2f} m | Y: {self.y:.2f} m | "
            f"angular_factor: {self.angular_factor:.3f}"
        )

def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
