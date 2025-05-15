from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_dir = get_package_share_directory('green_tracker_pkg')
    path_file = os.path.join(pkg_dir, 'config', 'configg.yaml')

    return LaunchDescription([
        # 🧭 Nodo de estimación de pose
        Node(
            package='green_tracker_pkg',
            executable='pose_estimatorg',
            name='pose_estimator',
            output='screen'
        ),

        # 🗺️ Nodo generador de trayectoria
        Node(
            package='green_tracker_pkg',
            executable='path_geng',
            name='path_generator',
            parameters=[{'path_file': path_file}],
            output='screen'
        ),

        # 🚦 Nodo de lógica semafórica con controlador
        Node(
            package='green_tracker_pkg',
            executable='semaforo_node',
            name='semaforo_node',
            output='screen'
        ),
        # 🛣️ Nodo de controlador de trayectoria
        Node(
            package='green_tracker_pkg',
            executable='closedgreencontroller',
            name='closed_loop_controller',
            output='screen',
            parameters=[
                {'Kp_lin': 1.0},
                {'Kp_ang': 2.0},
                {'MAX_V': 0.5},
                {'MAX_W': 1.0}
            ]
        ),
        Node(
            package='green_tracker_pkg',
            executable='cmd_vel_safe_publisher',
            name='cmd_vel_safe_publisher',
            output='screen'
        )
    ])
