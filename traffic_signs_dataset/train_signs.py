from ultralytics import YOLO

# Modelo más ligero
model = YOLO("yolov8n.pt")

# Entrenamiento optimizado para tu GPU
model.train(
    data="/home/felipe/Descargas/TrafficSignsLights.v1i.yolov8/data.yaml",
    epochs=50,
    imgsz=416,
    batch=4,
    name="senales_trafico_v1",
    device=0  # usa GPU si no se llena
)
