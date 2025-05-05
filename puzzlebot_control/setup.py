from setuptools import find_packages, setup

package_name = 'puzzlebot_control'

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
    maintainer='Mario Martinez',
    maintainer_email='mario.mtz@manchester-robotics.com',
    description='Puzzlebot controllers',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'open_loop_ctrl = puzzlebot_control.open_loop_ctrl:main',
            'pose_estimator = puzzlebot_control.pose_estimator:main',
            'closed_loop_ctrl = puzzlebot_control.closed_loop_ctrl:main',
        ],
    },
)
