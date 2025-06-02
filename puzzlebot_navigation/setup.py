from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'puzzlebot_navigation'

setup(
    name=package_name,
    version='1.0.0',  # Actualizada para consistencia con package.xml
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Incluir archivos launch si existen
        (os.path.join('share', package_name, 'launch'), 
         glob(os.path.join('launch', '*launch.[pxy]'))),
        # Incluir archivos de configuración si existen
        (os.path.join('share', package_name, 'config'), 
         glob(os.path.join('config', '*.yaml'))),
        # Incluir otros recursos si existen
        (os.path.join('share', package_name, 'worlds'), 
         glob(os.path.join('worlds', '*.world'))),
    ],
    install_requires=[
        'setuptools',
        # Dependencias de Python (no incluir opencv-python ya que usas python3-opencv del sistema)
        # 'opencv-python',  # Comentado porque usas python3-opencv del package.xml
        # 'numpy',          # Comentado porque usas python3-numpy del package.xml  
        # 'simple-pid',     # Comentado porque usas python3-simple-pid del package.xml
    ],
    zip_safe=True,
    maintainer='felipe',
    maintainer_email='felisuper13@hotmail.com',
    description='Navegación avanzada para PuzzleBot con seguimiento de líneas e intersecciones',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Nodos existentes
            'navigate_to_marker = puzzlebot_navigation.navigate_to_marker_node:main',
            'gstreamer = puzzlebot_navigation.gstreamer:main',
            'chess_flag_detector = puzzlebot_navigation.chess_flag_detector:main',
            'follow_traffic_chess = puzzlebot_navigation.follow_traffic_chess_node:main',
            'intersect_node = puzzlebot_navigation.intersect_node:main',

        ],
    },
)