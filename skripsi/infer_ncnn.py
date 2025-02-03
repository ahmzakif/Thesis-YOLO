import time
import glob
import tqdm
from pathlib import Path

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

from .visualize import vis, box_to_txt
from .classes import CLASSES
from .speed_tracker import SpeedTracker

class InferNCNN():
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.init_model()

    def init_model(self):
        model = YOLO(self.model_path, task='detect')
        return model

    def postprocess(self, outputs, conf_thres=0.6, iou_thres=0.6):
        detections = outputs[0].boxes
        boxes = []
        for i, box in enumerate(detections):
            conf = box.conf.item()
            if conf > conf_thres:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls.item())
                boxes.append([x1, y1, x2, y2, conf, cls])

        return boxes

    def visualize(self, img, boxes, conf_thr=0.6):
        if boxes:
            final_boxes = np.array(boxes)[:, :4]
            final_scores = np.array(boxes)[:, 4]
            final_cls_inds = np.array(boxes)[:, 5].astype(int)
            result_img = vis(img, final_boxes, final_scores, 
                             final_cls_inds, conf=conf_thr, class_names=CLASSES)
            return result_img
        return img

    def run_image(self, img_path, enable_vis=False, enable_write=False):
        img = cv2.imread(img_path)
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(img, verbose=False)
        boxes = self.postprocess(outputs)
        end_time = time.time()
        inference_time = end_time - start_time

        if enable_vis:
            result_img = self.visualize(img, boxes)
            plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            plt.show()
            print(f"Inference Time: {inference_time * 1000:.2f} ms")
            print(f"FPS: {1 / inference_time:.2f}")
            folder_name = Path(img_path).stem
            file_name = f'./outputs/prediction_{folder_name}.jpg'
            cv2.imwrite(file_name, result_img)
        if enable_write:
            if boxes:
                final_boxes = np.array(boxes)[:, :4]
                final_scores = np.array(boxes)[:, 4]
                final_cls_inds = np.array(boxes)[:, 5].astype(int)
                box_to_txt(img_path, final_boxes, final_scores, final_cls_inds, CLASSES, conf=0.6)

        return boxes

    def run_webcam(self, source, enable_vis=False):
        cap = cv2.VideoCapture(source)
        tracker = SpeedTracker()
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break
            if ret:
                start_time = time.time()
                with torch.no_grad():
                    outputs = self.model(frame, device=self.device, verbose=False)
                boxes = self.postprocess(outputs)
                end_time = time.time()
                inference_time = end_time - start_time

                if enable_vis:
                    result_img = self.visualize(frame, boxes, conf_thr=0.6)
                    cv2.putText(result_img, f'FPS: {1 / inference_time:.2f}',
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (213, 239, 255), 2)
                    cv2.putText(result_img, f'Inference Time: {inference_time * 1000:.2f} ms',
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 230, 240), 2)
                    tracker.speed_performance(start_time, end_time, limit_frame=500)
                    cv2.imshow('Webcam Inference', result_img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

    def run_batches(self, dir_path):
        for img_path in tqdm.tqdm(glob.glob(dir_path + '/*.jpg')):
            self.run_image(img_path, enable_vis=False, enable_write=True)
