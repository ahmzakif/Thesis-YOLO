from ultralytics import YOLO

model = YOLO("model\yolo11n_v4_bs32.pt")
model.export(format="ncnn")