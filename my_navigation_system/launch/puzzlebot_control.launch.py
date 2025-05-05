#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

def generate_launch_description():
    pkg = get_package_share_directory('my_navigation_system')
    urdf   = os.path.join(pkg, 'urdf', 'puzzlebot.urdf.xacro')
    config = os.path.join(pkg, 'config', 'controllers.yaml')

    return LaunchDescription([

        # 1) Publicar URDF en /robot_description
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': Command(['xacro ', urdf])
            }],
        ),

        # 2) Colocar el robot en Gazebo
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
              '--topic', 'robot_description',
              '--entity', 'puzzlebot1'
            ],
            output='screen'
        ),

        # 3) Iniciar ros2_control_node con tu YAML
        Node(
            package='controller_manager',
            executable='ros2_control_node',
            parameters=[config],
            output='screen'
        ),

        # 4) Levantar los controladores
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster']
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['diff_drive_controller']
        ),

        # 5) Teleop opcional
        Node(
            package='teleop_twist_keyboard',
            executable='teleop_twist_keyboard',
            name='teleop_twist_keyboard',
            prefix='xterm -e',
            output='screen'
        ),

    ])
