from setuptools import find_packages, setup

package_name = 'te3002b_sim_bridge'

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
    description='Bridge entre ROS 2 y simulador TE3002B',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rpc_image_node = te3002b_sim_bridge.rpc_image_node:main',
        ],
    },
)