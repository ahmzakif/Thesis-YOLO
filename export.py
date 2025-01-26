from ultralytics import YOLO

model = YOLO("model\yolo11n.pt")
model.export(format="ncnn")