# 🤖 Puzzlebot ROS 2 Workspace

<div align="center">
  
![Puzzlebot Logo](https://via.placeholder.com/150x150)

**Advanced mobile robotics platform for education and research**

[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue)](https://docs.ros.org/en/humble/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()

</div>

📋 Overview
This repository contains the complete ROS 2 workspace for the Puzzlebot, a differential mobile robot platform designed for advanced control, navigation, and perception practices. The workspace integrates all packages, drivers, and configurations developed for both simulation and real hardware execution (Jetson + encoders).
✨ Key Features

Modular Architecture: Clean separation of components for easy extension and modification
Hardware Integration: Seamless connection with Jetson-based hardware and encoders
Advanced Control: PID controllers for precise movement and trajectory following
Computer Vision: Real-time image processing for object tracking and traffic light detection
Navigation System: Waypoint navigation with path planning capabilities
Simulation Support: Full compatibility with Gazebo for testing before hardware deployment

🗂️ Repository Structure
DirectoryDescription.vscode/Development environment configurations (extensions, launch.json, etc.)build/, install/, log/Automatically generated colcon build directoriesgreen_tracker_pkg/Computer vision package for green object tracking and traffic light detectionmi_control_puzzlebot/Custom PID controllers and closed-loop control nodesmy_navigation_system/Complete navigation system with waypoints and control integrationopen_loop_control/Open-loop control modules for basic testingpuzzlebot_control/Modular ROS 2 controllers and launchers for Puzzlebotpuzzlebot_navigation/Line follower package with vision-based line detection and tracking algorithmspuzzlebot_ros/Base package with URDF models, configurations and integrated sensorspuzzlebot_teleop/Teleoperation package for manual robot control
🛠️ Installation
Prerequisites

ROS 2 Humble (on Ubuntu 22.04)
Python 3.10+
OpenCV 4.x

Setup


<div align="center">
  
📫 **Contact**: [A01705893@tec.mx]

</div>
