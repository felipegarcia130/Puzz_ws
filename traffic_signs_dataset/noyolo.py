import cv2
import numpy as np

class TrafficSignDetector:
    def __init__(self):
        # Rangos de color en HSV para cada tipo de señal
        self.color_ranges = {
            'red_signs': {
                'lower1': np.array([0, 120, 70]),
                'upper1': np.array([10, 255, 255]),
                'lower2': np.array([170, 120, 70]),
                'upper2': np.array([180, 255, 255])
            },
            'blue_signs': {
                'lower': np.array([100, 150, 0]),
                'upper': np.array([130, 255, 255])
            }
        }
    
    def detect_signs(self, frame):
        """Detecta señales en el frame de la cámara"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detected_signs = []
        
        # Detectar señales rojas (STOP, GIVE WAY)
        red_signs = self.detect_red_signs(hsv, frame)
        detected_signs.extend(red_signs)
        
        # Detectar señales azules (direccionales)
        blue_signs = self.detect_blue_signs(hsv, frame)
        detected_signs.extend(blue_signs)
        
        return detected_signs
    
    def detect_red_signs(self, hsv, frame):
        """Detecta señales rojas"""
        # Crear máscara para colores rojos (dos rangos por el wrap del rojo en HSV)
        mask1 = cv2.inRange(hsv, self.color_ranges['red_signs']['lower1'], 
                           self.color_ranges['red_signs']['upper1'])
        mask2 = cv2.inRange(hsv, self.color_ranges['red_signs']['lower2'], 
                           self.color_ranges['red_signs']['upper2'])
        red_mask = mask1 + mask2
        
        # Limpiar la máscara
        kernel = np.ones((5,5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filtrar por área mínima
                # Aproximar el contorno
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                x, y, w, h = cv2.boundingRect(contour)
                
                # Clasificar por forma
                if len(approx) == 8 and abs(w-h) < 20:  # Octágono (STOP)
                    detected.append({
                        'type': 'STOP',
                        'bbox': (x, y, w, h),
                        'confidence': self.calculate_confidence(contour, 'octagon')
                    })
                elif len(approx) == 3:  # Triángulo (GIVE WAY)
                    detected.append({
                        'type': 'GIVE_WAY',
                        'bbox': (x, y, w, h),
                        'confidence': self.calculate_confidence(contour, 'triangle')
                    })
        
        return detected
    
    def detect_blue_signs(self, hsv, frame):
        """Detecta señales azules direccionales"""
        blue_mask = cv2.inRange(hsv, self.color_ranges['blue_signs']['lower'], 
                               self.color_ranges['blue_signs']['upper'])
        
        # Limpiar la máscara
        kernel = np.ones((5,5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Extraer la región de interés para análisis de flecha
                roi = frame[y:y+h, x:x+w]
                direction = self.detect_arrow_direction(roi)
                
                detected.append({
                    'type': f'ARROW_{direction}',
                    'bbox': (x, y, w, h),
                    'confidence': 0.8
                })
        
        return detected
    
    def detect_arrow_direction(self, roi):
        """Detecta la dirección de la flecha en la ROI"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Aplicar threshold para obtener la flecha blanca
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Encontrar contornos de la flecha
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 'UNKNOWN'
        
        # Tomar el contorno más grande (debería ser la flecha)
        arrow_contour = max(contours, key=cv2.contourArea)
        
        # Calcular momentos para encontrar el centroide
        M = cv2.moments(arrow_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # Analizar la distribución de píxeles para determinar dirección
            h, w = roi.shape[:2]
            center_x = w // 2
            center_y = h // 2
            
            if cx < center_x - 10:
                return 'LEFT'
            elif cx > center_x + 10:
                return 'RIGHT'
            elif cy < center_y - 10:
                return 'UP'
            elif cy > center_y + 10:
                return 'DOWN'
        
        return 'STRAIGHT'
    
    def calculate_confidence(self, contour, expected_shape):
        """Calcula confianza basada en qué tan bien coincide la forma"""
        if expected_shape == 'octagon':
            # Para octágono, verificar que tenga aproximadamente 8 lados
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            return max(0.5, min(1.0, 8.0 / max(len(approx), 8)))
        elif expected_shape == 'triangle':
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            return max(0.5, min(1.0, 3.0 / max(len(approx), 3)))
        return 0.7
    
    def draw_detections(self, frame, detections):
        """Dibuja las detecciones en el frame"""
        for detection in detections:
            x, y, w, h = detection['bbox']
            
            # Color del bounding box según el tipo
            if 'STOP' in detection['type']:
                color = (0, 0, 255)  # Rojo
            elif 'GIVE_WAY' in detection['type']:
                color = (0, 165, 255)  # Naranja
            elif 'ARROW' in detection['type']:
                color = (255, 0, 0)  # Azul
            else:
                color = (0, 255, 0)  # Verde
            
            # Dibujar bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Dibujar etiqueta
            label = f"{detection['type']} ({detection['confidence']:.2f})"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 2)
        
        return frame

# Ejemplo de uso con la cámara de Jetson
def main():
    detector = TrafficSignDetector()
    
    # Inicializar cámara (ajusta según tu configuración)
    cap = cv2.VideoCapture(0)  # o el índice correcto para tu cámara
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detectar señales
        detections = detector.detect_signs(frame)
        
        # Dibujar detecciones
        result_frame = detector.draw_detections(frame, detections)
        
        # Mostrar resultado
        cv2.imshow('Traffic Sign Detection', result_frame)
        
        # Imprimir detecciones
        if detections:
            print(f"Detected {len(detections)} signs:")
            for det in detections:
                print(f"  - {det['type']} at {det['bbox']} (conf: {det['confidence']:.2f})")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()