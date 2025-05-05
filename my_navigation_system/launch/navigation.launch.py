from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_path = get_package_share_directory('my_navigation_system')
    path_yaml = os.path.join(pkg_path, 'config', 'path.yaml')

    return LaunchDescription([
        Node(
            package='my_navigation_system',
            executable='path_generator_node',
            name='path_generator',
            output='screen',
            parameters=[{'path_file': path_yaml}]
        ),
        Node(
            package='my_navigation_system',
            executable='controller_node',
            name='controller',
            output='screen'
        )
    ])
