from setuptools import find_packages, setup

package_name = 'puzzlebot_teleop'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/teleop_launch.py']),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='felipe',
    maintainer_email='felipe@todo.todo',
    description='Teleoperación del Puzzlebot con un control de PS4 usando ROS 2 y joy_node',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'teleop_joystick_node = puzzlebot_teleop.teleop_joystick_node:main',
        ],
    },
)
