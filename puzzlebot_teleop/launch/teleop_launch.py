from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Nodo joy_node para leer joystick (control PS4)
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            output='screen',
            parameters=[{
                'dev': '/dev/input/js0',          # Ruta del dispositivo (USB)
                'deadzone': 0.05,                 # Zona muerta para el stick
                'autorepeat_rate': 20.0           # Frecuencia de repeticiones si mantienes presionado
            }]
        ),

        # Nodo tuyo que convierte entradas de /joy a /cmd_vel
        Node(
            package='puzzlebot_teleop',
            executable='teleop_joystick_node',
            name='teleop_joystick_node',
            output='screen'
        )
    ])
