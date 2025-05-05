from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'green_tracker_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='felipe',
    maintainer_email='felipe@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'semaforo_node=green_tracker_pkg.semaforo_node:main',
            'green_tracker_node=green_tracker_pkg.green_tracker_node:main',
            'pose_estimatorg=green_tracker_pkg.pose_estimatorg:main',
            'path_geng=green_tracker_pkg.path_geng:main',
        ],
    },
)
