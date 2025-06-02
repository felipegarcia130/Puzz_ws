from ultralytics import YOLO
import cv2
import time

# Cargar modelo
model = YOLO("/home/felipe/puzz_ws/src/traffic_signs_dataset/runs/train/traffic_signs_v8/weights/best.pt")

# CLASES CORRECTAS (confirmadas que funcionan)
CLASSES = {
    0: "CONSTRUCTION",
    1: "ROUNDABOUT", 
    2: "STOP",
    3: "GO STRAIGHT",
    4: "TURN LEFT",
    5: "TURN RIGHT"
}

# Colores para cada clase (BGR)
COLORS = {
    0: (0, 255, 0),     # CONSTRUCTION - Verde
    1: (255, 0, 255),   # ROUNDABOUT - Magenta
    2: (0, 0, 255),     # STOP - Rojo brillante
    3: (255, 255, 0),   # GO STRAIGHT - Cian
    4: (255, 0, 0),     # TURN LEFT - Azul
    5: (0, 255, 255)    # TURN RIGHT - Amarillo
}

cv2.destroyAllWindows()
time.sleep(0.1)

WINDOW_NAME = "🚦 Detector de Señales - FUNCIONANDO"

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ No se pudo abrir la cámara.")
    exit()

# Configurar resolución óptima
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

print("🎉 ¡DETECTOR ACTIVO Y FUNCIONANDO!")
print("🔍 Señales que puede detectar:")
for id, name in CLASSES.items():
    print(f"   {id}: {name}")
print("\n💡 CONSEJOS:")
print("   📏 Mantén señales a 1-2 metros")
print("   💡 Buena iluminación")
print("   🎯 Centra la señal en pantalla")
print("   ⏱️ Mantén la señal quieta 1-2 segundos")
print("\n🎮 Controles: 'q' = salir | 's' = captura | 'r' = resetear\n")

try:
    frame_count = 0
    last_detection_time = time.time()
    detection_history = []
    last_confident_detection = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ No se pudo leer frame")
            continue

        frame_count += 1
        h, w = frame.shape[:2]
        
        # Detección con configuración optimizada
        results = model.predict(frame, 
                               imgsz=640, 
                               conf=0.3,  # Confianza moderada
                               verbose=False,
                               device='cpu')[0]
        
        detections = results.boxes if results.boxes is not None else []
        annotated = frame.copy()
        
        current_time = time.time()
        current_detections = []
        
        # Procesar detecciones
        for box in detections:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = CLASSES.get(cls_id, f"Clase {cls_id}")
            color = COLORS.get(cls_id, (128, 128, 128))
            
            # Calcular tamaño de la detección
            box_area = (x2 - x1) * (y2 - y1)
            screen_area = w * h
            size_ratio = box_area / screen_area
            
            # Grosor según confianza y tamaño
            thickness = max(2, int(conf * 4))
            
            # Dibujar caja
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Etiqueta con información detallada
            label_text = f"{label} {conf*100:.0f}%"
            if size_ratio > 0.02:  # Si la señal es lo suficientemente grande
                label_text += " ✓"
            
            # Fondo para texto
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, (x1, y1-text_h-15), (x1+text_w+10, y1), color, -1)
            cv2.putText(annotated, label_text, (x1+5, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Registrar detección
            detection_info = f"{label} ({conf*100:.0f}%)"
            current_detections.append(detection_info)
            
            # Detección confiable
            if conf > 0.6 and size_ratio > 0.02:
                last_confident_detection = {
                    'label': label,
                    'conf': conf,
                    'time': current_time
                }
        
        # Imprimir detecciones cada 2 segundos
        if current_detections and (current_time - last_detection_time) >= 2.0:
            print(f"🔎 DETECTANDO: {', '.join(current_detections)}")
            detection_history.extend(current_detections)
            last_detection_time = current_time
        
        # Información en pantalla
        info_y = 30
        cv2.putText(annotated, f"Frames: {frame_count} | Detecciones: {len(detections)}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Mostrar última detección confiable
        if last_confident_detection and (current_time - last_confident_detection['time']) < 5:
            info_y += 30
            confident_text = f"CONFIRMADO: {last_confident_detection['label']} ({last_confident_detection['conf']*100:.0f}%)"
            cv2.putText(annotated, confident_text, (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
        
        # Zona de enfoque
        center_x, center_y = w//2, h//2
        zone_size = min(w, h) // 4
        cv2.rectangle(annotated, 
                     (center_x - zone_size//2, center_y - zone_size//2),
                     (center_x + zone_size//2, center_y + zone_size//2),
                     (0, 255, 0), 2)
        
        # Instrucciones
        cv2.putText(annotated, "q=salir | s=captura | r=reset", 
                   (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, annotated)

        # Controles
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):  # Capturar
            timestamp = int(time.time())
            filename = f"captura_deteccion_{timestamp}.jpg"
            cv2.imwrite(filename, annotated)
            print(f"📸 Captura guardada: {filename}")
        elif key == ord('r'):  # Reset
            detection_history.clear()
            last_confident_detection = None
            print("🔄 Historial reseteado")

        time.sleep(0.03)

except KeyboardInterrupt:
    print("\n🛑 Detenido por usuario")
except Exception as e:
    print(f"❌ Error: {e}")
finally:
    print("\n📊 RESUMEN DE SESIÓN:")
    print(f"   Frames procesados: {frame_count}")
    
    if detection_history:
        unique_detections = list(set(detection_history))
        print(f"   ✅ Señales detectadas: {', '.join(unique_detections)}")
        print(f"   📈 Total detecciones: {len(detection_history)}")
    else:
        print("   ℹ️ No se registraron detecciones en esta sesión")
    
    print("🔄 Cerrando...")
    cap.release()
    cv2.destroyAllWindows()
    
    for i in range(5):
        cv2.waitKey(1)
        time.sleep(0.1)
    
    print("✅ ¡Detector cerrado correctamente!")