# 🤖 Puzzlebot ROS 2 Workspace

<div align="center">
  <img src=https://github.com/user-attachments/assets/6c20fec5-3883-4be3-b402-44fbccc083f6>
  <p><em>Figura 1. Puzzlebot armado.</em></p>
</div>

**Advanced mobile robotics platform for education and research**

[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue)](https://docs.ros.org/en/humble/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()

## 📋 Overview

This repository contains the complete ROS 2 workspace for the **Puzzlebot**, a differential mobile robot platform designed for advanced control, navigation, and perception practices. The workspace integrates all packages, drivers, and configurations developed for both simulation and real hardware execution (Jetson + encoders).

## ✨ Key Features

- **Modular Architecture**: Clean separation of components for easy extension and modification
- **Hardware Integration**: Seamless connection with Jetson-based hardware and encoders
- **Advanced Control**: PID controllers for precise movement and trajectory following
- **Computer Vision**: Real-time image processing for object tracking and traffic light detection
- **Navigation System**: Waypoint navigation with path planning capabilities
- **Simulation Support**: Full compatibility with Gazebo and TE3002B track simulator for testing before hardware deployment
- **Person Following**: Advanced computer vision capabilities for human detection and following

## 🗂️ Repository Structure

| Directory | Description |
|-----------|-------------|
| `.vscode/` | Development environment configurations (extensions, launch.json, etc.) |
| `build/`, `install/`, `log/` | Automatically generated colcon build directories |
| `green_tracker_pkg/` | Computer vision package for green object tracking and traffic light detection |
| `mi_control_puzzlebot/` | Custom PID controllers and closed-loop control nodes |
| `my_navigation_system/` | Complete navigation system with waypoints and control integration |
| `open_loop_control/` | Open-loop control modules for basic testing |
| `pb_camera_bridge/` | Camera bridge package for Puzzlebot camera integration |
| `person_follower/` | Advanced computer vision package for human detection and following |
| `puzzlebot_control/` | Modular ROS 2 controllers and launchers for Puzzlebot |
| `puzzlebot_navigation/` | Line follower package with vision-based line detection and tracking algorithms |
| `puzzlebot_ros/` | Base package with URDF models, configurations and integrated sensors |
| `puzzlebot_teleop/` | Teleoperation package for manual robot control |
| `te3002b_sim_bridge/` | Simulation bridge for TE3002B track simulator integration |

## 🛠️ Installation

### Prerequisites
- ROS 2 Humble (on Ubuntu 22.04)
- Python 3.10+
- OpenCV 4.x

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/puzz_ws.git
cd puzz_ws

# Install dependencies
rosdep update
rosdep install --from-paths src --ignore-src -r -y

# Build the workspace
colcon build
source install/setup.bash
```

## 🚀 Usage

### Running on Physical Robot
```bash
# Connect to Puzzlebot
ssh puzzlebot@192.168.137.xxx

# Set environment variables
export ROS_DOMAIN_ID=0
export ROS_IP=192.168.137.xxx

# Launch micro-ROS agent
ros2 launch puzzlebot_ros micro_ros_agent.launch.py

# In a new terminal, run teleop for manual control
ros2 run teleop_twist_keyboard teleop_twist_keyboard

# For the camera
#(In jetson)
ros2 run green_tracker_pkg jetson_camera_node
#Or:
#(In Jetson)
./start_camera.sh
#(In Laptop)
ros2 run puzzlebot_navigation gstreamer

# For the line follower
ros2 run puzzlebot_navigation navigate_to_marker
ros2 run puzzlebot_navigation follow_line_with_traffic_node

# For person following
ros2 run person_follower person_detection_node
```

### Running in Simulation
```bash
# Launch TE3002B track simulation
ros2 run te3002b_sim_bridge track_simulator

# Launch simulation using rpc_image_node.py
ros2 run [package_name] rpc_image_node.py

# Launch navigation stack
ros2 launch my_navigation_system navigation.launch.py

# Camera bridge for simulation
ros2 run pb_camera_bridge camera_bridge_node
```

### Creating New Packages
```bash
# Create a new Python package
ros2 pkg create --build-type ament_python package_name --node-name node_name

# Build specific packages
colcon build --packages-select package_name
```

## 📊 Implemented Modules

### Control Systems
- Linear and angular PID controllers
- Velocity and position control
- Trajectory tracking

### Computer Vision
- Real-time green object detection and tracking
- Traffic light state recognition
- Human detection and following capabilities
- Camera calibration tools
- Camera bridge integration for hardware and simulation

### Navigation
- Waypoint-based navigation
- Path planning and following
- Line following with traffic light recognition
- Odometry estimation using encoders

### Simulation
- TE3002B track simulator integration
- Hardware-in-the-loop testing capabilities
- Camera simulation bridge

### Visualization
- Custom RViz configurations
- Real-time sensor data plotting
- Path visualization

## 🎯 Specialized Features

### Line Following System
The `puzzlebot_navigation` package implements advanced line following capabilities using computer vision algorithms. It features real-time line detection, path tracking, and traffic light recognition, enabling autonomous navigation along predefined paths with obstacle and traffic signal awareness.

### Person Following System
The `person_follower` package provides advanced computer vision capabilities for detecting and following humans, making the Puzzlebot suitable for social robotics applications and human-robot interaction research.

### TE3002B Track Simulator
The `te3002b_sim_bridge` package enables seamless integration with the TE3002B track simulator, providing a realistic testing environment that mirrors real-world track conditions before deploying to physical hardware.

### Camera Integration
Multiple camera packages (`pb_camera_bridge`, `green_tracker_pkg`) provide comprehensive camera support for both hardware and simulation environments, enabling robust computer vision applications.

## 📚 Documentation

Complete documentation is available in the `docs/` directory, including:
- Hardware setup guide
- Software architecture overview
- API reference
- Testing procedures
- Simulation setup guide

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Team

| Member | Role |
|--------|------|
| **Felipe de Jesús García García** | ROS Integration Lead |
| **Samuel Cabrera** | Coppelia Sim Simulation Specialist |
| **José Luis Urquieta** | Computer Vision Logic |
| **Uriel Lemuz** | 3D Modeling and Construction |
| **Santiago Lopez** | Documentation |

## 👏 Acknowledgements

- ROS 2 Community
- Robotics Lab at [Your Institution]
- Contributors and maintainers

---

<div align="center">
  
📫 **Contact**: [A01705893@tec.com]

</div>
