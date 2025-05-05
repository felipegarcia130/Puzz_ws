# puzz_ws

Repositorio principal de desarrollo para el **Puzzlebot**, un robot móvil diferencial usado en prácticas avanzadas de control, navegación y percepción utilizando **ROS 2**.

Este workspace agrupa todos los paquetes, controladores y configuraciones desarrolladas a lo largo del tiempo para uso tanto en simulación como en ejecución real con hardware (Jetson + encoders).

---

## 📁 Estructura del repositorio

| Carpeta                  | Descripción                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `.vscode/`               | Configuraciones del entorno de desarrollo (extensiones, launch.json, etc.) |
| `build/`, `install/`, `log/` | Directorios generados automáticamente por colcon                        |
| `green_tracker_pkg/`     | Paquete de seguimiento visual de objetos verdes y semáforo (visión artificial)         |
| `mi_control_puzzlebot/`  | Controladores PID personalizados y nodos de control cerrado                 |
| `my_navigation_system/`  | Sistema completo de navegación con waypoints y control                      |
| `open_loop_control/`     | Control en lazo abierto para pruebas simples                                |
| `puzzlebot_control/`     | Controladores modulares y lanzadores ROS 2 para el Puzzlebot                |
| `puzzlebot_ros/`         | Paquete base con URDF, configuraciones y sensores integrados                |

---

## 🧠 Características implementadas

- Control PID lineal y angular con ROS 2
- Seguimiento visual con procesamiento de imágenes en tiempo real
- Navegación por waypoints y rutas predefinidas
- Estimación de odometría usando encoders
- Visualización en RViz
- Configuración para uso en **Jetson** y **ROS 2 Humble**

---

## 🚀 Uso básico

```bash
ssh puzzlebot@192.168.137.???
export ROS_DOMAIN_ID=0
export ROS_IP=192.168.137.???
ros2 launch puzzlebot_ros micro_ros_agent.launch.py
ros2 run teleop_twist_keyboard teleop_twist_keyboard
cd ~/ros2_ws
ros2 pkg create --build-type ament_python package_name node_name
colcon build --packages-select name

