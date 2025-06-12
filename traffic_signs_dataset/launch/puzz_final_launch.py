#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='puzzlebot_navigation',
            executable='yolo_sign_detector_node',
            name='yolo_detector',
            parameters=[{
                'model_path': '/home/felipe/puzz_ws/src/traffic_signs_dataset/best.pt'
            }],
            output='screen'
        ),
        Node(
            package='puzzlebot_navigation',
            executable='intersection_detector_node',
            name='intersection_detector',
            output='screen'
        ),
        Node(
            package='puzzlebot_navigation',
            executable='stoplight_detector_node',
            name='stoplight_detector',
            output='screen'
        ),
        Node(
            package='puzzlebot_navigation',
            executable='flag_detector_node',
            name='flag_detector',
            output='screen'
        ),
        Node(
            package='puzzlebot_navigation',
            executable='line_follower_node',
            name='line_follower',
            output='screen'
        ),
        Node(
            package='puzzlebot_navigation',
            executable='track_navigator_node',
            name='track_navigator',
            output='screen'
        )
    ])
