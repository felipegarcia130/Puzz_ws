#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg = get_package_share_directory('my_navigation_system')
    world = os.path.join(pkg, 'worlds', 'world_empty.sdf') 

    return LaunchDescription([
        Node(
            package='gazebo_ros',
            executable='gzserver',
            name='gazebo_server',
            output='screen',
            arguments=['--verbose', world]
        ),
        Node(
            package='gazebo_ros',
            executable='gzclient',
            name='gazebo_client',
            output='screen'
        ),
    ])
