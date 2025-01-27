from ultralytics import YOLO

model = YOLO("model/best11n.pt")
model.export(format="ncnn")
# model.info()