from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_dir = get_package_share_directory('my_navigation_system')
    return LaunchDescription([
        Node(
            package='my_navigation_system',
            executable='controller_node',
            name='controller',
            output='screen'
        ),
        Node(
            package='my_navigation_system',
            executable='trajectory_gui_node',
            name='trajectory_gui',
            output='screen'
        )
    ])
