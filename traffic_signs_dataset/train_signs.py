from ultralytics import YOLO

# Carga del modelo base (tiny)
model = YOLO("yolov8n.pt")

# Entrenamiento
model.train(
    data="data/data.yaml",   # Ruta relativa dentro del paquete
    epochs=50,
    imgsz=640,
    batch=8,
    name="traffic_signs_v8",
    project="runs/train"
)
