from setuptools import find_packages, setup

package_name = 'puzzlebot_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'opencv-python',
        'numpy',
        'simple-pid',
    ],
    zip_safe=True,
    maintainer='felipe',
    maintainer_email='felisuper13@hotmail.com',
    description='Seguidor de línea para el PuzzleBot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'navigate_to_marker = puzzlebot_navigation.navigate_to_marker_node:main',
            'follow_line_with_traffic = puzzlebot_navigation.follow_line_with_traffic_node:main',
            'gstreamer = puzzlebot_navigation.gstreamer:main',
        ],
    },
)

