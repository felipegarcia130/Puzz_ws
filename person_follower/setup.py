from setuptools import find_packages, setup

package_name = 'person_follower'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='felipe',
    maintainer_email='felisuper13@hotmail.com',
    description='Detecta y sigue personas con YOLO + PID',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_person_tracker_node = person_follower.yolo_person_tracker_node:main',
            'person_follower_controller_node = person_follower.person_follower_controller_node:main',
        ],
    },
)
