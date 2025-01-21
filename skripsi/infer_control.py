import glob
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from ultralytics import YOLO

from .classes import CLASSES
from .speed_tracker import SpeedTracker
from .controller import *
from .visualize import vis


class InferTorch():
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.init_model()
        self.command = Controller()

    def init_model(self):
        model = YOLO(self.model_path)
        return model

    def preprocess(self, img, img_size=640):
        # Resize and normalize the input image
        h, w, _ = img.shape
        r = img_size / max(h, w)
        new_w, new_h = int(w * r), int(h * r)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded_img = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
        padded_img[:new_h, :new_w, :] = resized_img

        img = padded_img.astype(np.float32)
        img = img / 255.0  # Normalize to [0, 1]
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return torch.from_numpy(img).to(self.device), r

    def postprocess(self, outputs, ratio, conf_thres=0.5, iou_thres=0.45):
        detections = outputs[0].boxes  
        boxes = []
        
        for i, box in enumerate(detections):  
            conf = box.conf.item()  
            if conf > conf_thres:  
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  
                cls = int(box.cls.item())  
                boxes.append([x1 / ratio, y1 / ratio, x2 / ratio, y2 / ratio, conf, cls])

        return boxes


    def visualize(self, img, boxes, conf_thr=0.5):
        if boxes:
            final_boxes = np.array(boxes)[:, :4]
            final_scores = np.array(boxes)[:, 4]
            final_cls_inds = np.array(boxes)[:, 5].astype(int)
            result_img = vis(img, final_boxes, final_scores, final_cls_inds, conf=conf_thr, class_names=CLASSES)
            return result_img
        return img
    
    def run(self, source, enable_vis=False):
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        detected = ''
        start_time = 0

        while True:
            ret, frame = cap.read()
            tracker = SpeedTracker()

            if not ret:
                print("Failed to capture frame.")
                break

            # Preprocess frame
            img_tensor, ratio = self.preprocess(frame)
            prev_time = time.time()

            # Run inference
            with torch.no_grad():
                outputs = self.model(img_tensor)

            # Postprocess results
            boxes = self.postprocess(outputs, ratio)

            current_time = time.time()
            inference_time = current_time - prev_time
            
            if enable_vis and len(boxes) > 0:
                box_coords = [box[:4] for box in boxes]
                scores = [box[4] for box in boxes]
                cls_ids = [int(box[5]) for box in boxes]
                frame = vis(frame, box_coords, scores, cls_ids, conf=0.5, class_names=CLASSES)

            if not boxes:
                if not self.command.turning:
                    print("No object detected.")
                    self.command.servo_reset()
            else:
                for box in boxes:
                    cls_name = CLASSES[int(box[5])]

                    if not self.command.turning and (((time.time() - start_time) % 60) > 1) and detected != '':
                        if detected == 'Metal':
                            print("Metal detected.")
                            self.command.servo_control('metal')

                        elif detected == 'Plastic':
                            print("Plastic detected.")
                            self.servo.servo_control('plastic') 

                    if cls_name == 'Metal' and detected != 'Metal':
                        start_time = time.time()
                        detected = 'Metal'
                    
                    elif cls_name == 'Plastic' and detected != 'Plastic':
                        start_time = time.time()
                        detected = 'Plastic'
                        
            if enable_vis:
                if boxes:
                    result_img = self.visualize(frame, boxes, conf_thr=0.5)
                else:
                    result_img = frame

                cv2.putText(result_img, f'FPS: {1 / inference_time:.2f}', 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (213, 239, 255), 2)
                cv2.putText(result_img, f'Inference Time: {inference_time * 1000:.2f} ms', 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 230, 240), 2)

                tracker.speed_performance(prev_time, current_time, limit_frame=500)
                cv2.imshow("Webcam Inference", result_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                self.command.servo_reset()
                self.command.send_message('R')
                break

        cap.release()
        cv2.destroyAllWindows()