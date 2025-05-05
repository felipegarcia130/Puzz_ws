from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config_path = os.path.join(
        get_package_share_directory('my_navigation_system'),
        'config',
        'path.yaml'
    )

    return LaunchDescription([
        Node(
            package='my_navigation_system',
            executable='pose_estimate',
            name='pose_estimate',
            output='screen'
        ),
        Node(
            package='my_navigation_system',
            executable='path_generator_node',
            name='path_generator',
            output='screen',
            parameters=[{'path_file': config_path}]
        ),
        Node(
            package='my_navigation_system',
            executable='closed_loop_controller',
            name='closed_loop_controller',
            output='screen',
            parameters=[
                {'Kp_lin': 1.2},
                {'Ki_lin': 0.03},
                {'Kd_lin': 0.1},
                {'Kp_ang': 2.5},
                {'Ki_ang': 0.0},
                {'Kd_ang': 0.1},
                {'MAX_V': 0.2},
                {'MAX_W': 1.0},
                {'TOL_POS': 0.04},
                {'TOL_ANG': 0.1}
            ]
            
        )
    ])
