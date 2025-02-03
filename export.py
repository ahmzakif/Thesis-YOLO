from ultralytics import YOLO

model = YOLO("model/best32.pt")
model.export(format="ncnn")
