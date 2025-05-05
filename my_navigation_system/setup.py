from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'my_navigation_system'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
            (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='puzzlebot',
    maintainer_email='puzzlebot@todo.todo',
    description='Sistema de navegación con generación de trayectorias y control',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'path_generator_node = my_navigation_system.path_generator_node:main',
            'controller_node = my_navigation_system.controller_node:main',
            'closed_loop_controller = my_navigation_system.closed_loop_controller:main',
            'pose_estimate = my_navigation_system.pose_estimate:main',
            'rviz_simulator_node = my_navigation_system.sim_rviz_node:main',
            'trajectory_gui_node = my_navigation_system.trajectory_gui_node:main'

        ],
    },
)
