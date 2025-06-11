from setuptools import find_packages, setup

package_name = 'traffic_signs_dataset'

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
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detect_traffic_signs = traffic_signs_dataset.detect_traffic_signs:main',
            'line_intersect_signs = traffic_signs_dataset.line_intersect_signs:main',
            'line_intersect_traffic_signs = traffic_signs_dataset.line_intersect_traffic_signs:main',

        ],
    },
)
