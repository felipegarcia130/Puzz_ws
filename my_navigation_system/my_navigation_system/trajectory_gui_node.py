import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QHBoxLayout
import sys
import math
import tf_transformations
import time

class CoordinateSender(Node):
    def __init__(self):
        super().__init__('coordinate_sender')
        self.pub = self.create_publisher(PoseStamped, 'goal_pose', 10)

    def send_point(self, x, y):
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0.0

        # Default orientation (0 rad)
        q = tf_transformations.quaternion_from_euler(0, 0, 0)
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]

        self.pub.publish(msg)
        self.get_logger().info(f'📤 Punto enviado: ({x}, {y})')

class CoordinateGUI(QWidget):
    def __init__(self, ros_node):
        super().__init__()
        self.setWindowTitle('🧭 Ingresar Coordenadas')
        self.ros_node = ros_node
        self.layout = QVBoxLayout()

        self.coord_inputs = []

        # Ingreso del número de coordenadas
        self.num_label = QLabel('¿Cuántos puntos deseas ingresar?')
        self.num_input = QLineEdit()
        self.start_button = QPushButton('Continuar')
        self.start_button.clicked.connect(self.create_coordinate_fields)

        self.layout.addWidget(self.num_label)
        self.layout.addWidget(self.num_input)
        self.layout.addWidget(self.start_button)

        self.setLayout(self.layout)

    def create_coordinate_fields(self):
        try:
            n = int(self.num_input.text())
        except ValueError:
            self.num_label.setText('❌ Número inválido. Intenta de nuevo.')
            return

        # Limpiar layout
        for i in reversed(range(self.layout.count())):
            widget = self.layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        self.coord_inputs = []
        for i in range(n):
            row = QHBoxLayout()
            x_input = QLineEdit()
            y_input = QLineEdit()
            x_input.setPlaceholderText(f'X{i+1}')
            y_input.setPlaceholderText(f'Y{i+1}')
            row.addWidget(QLabel(f'Punto {i+1}:'))
            row.addWidget(x_input)
            row.addWidget(y_input)
            self.layout.addLayout(row)
            self.coord_inputs.append((x_input, y_input))

        self.send_button = QPushButton('Iniciar trayecto')
        self.send_button.clicked.connect(self.send_coordinates)
        self.layout.addWidget(self.send_button)

    def send_coordinates(self):
        for x_input, y_input in self.coord_inputs:
            try:
                x = float(x_input.text())
                y = float(y_input.text())
                self.ros_node.send_point(x, y)
                time.sleep(0.2)  # Pequeña pausa para no saturar
            except ValueError:
                self.num_label.setText('❌ Error en las coordenadas. Verifica que todos los campos sean numéricos.')
                return
        self.num_label.setText('✅ Coordenadas enviadas correctamente.')


def main():
    rclpy.init()
    ros_node = CoordinateSender()

    app = QApplication(sys.argv)
    gui = CoordinateGUI(ros_node)
    gui.show()

    timer = ros_node.create_timer(0.1, lambda: None)  # Necesario para que rclpy procese

    while app and not rclpy.ok():
        app.processEvents()

    rclpy.shutdown()

if __name__ == '__main__':
    main()
