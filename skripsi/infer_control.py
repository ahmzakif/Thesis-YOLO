import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO

from .controller import *
from .visualize import vis
from .classes import CLASSES

class InferNCNN():
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.init_model()
        self.command = Controller()

    def init_model(self):
        model = YOLO(self.model_path, task='detect')
        return model

    def postprocess(self, outputs, conf_thres=0.5, iou_thres=0.45):
        detections = outputs[0].boxes
        boxes = []
        for i, box in enumerate(detections):
            conf = box.conf.item()
            if conf > conf_thres:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls.item())
                boxes.append([x1, y1, x2, y2, conf, cls])

        return boxes

    def visualize(self, img, boxes, conf_thr=0.5):
        if boxes:
            final_boxes = np.array(boxes)[:, :4]
            final_scores = np.array(boxes)[:, 4]
            final_cls_inds = np.array(boxes)[:, 5].astype(int)
            result_img = vis(img, final_boxes, final_scores, 
                             final_cls_inds, conf=conf_thr, class_names=CLASSES)
            return result_img
        return img
    
    def run(self, source, enable_vis=False):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        detected = ''
        start_time = 0
        self.command.send_message('I')
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(frame)
            boxes = self.postprocess(outputs)
            end_time = time.time()
            inference_time = end_time - start_time
            
            if enable_vis and len(boxes) > 0:
                box_coords = [box[:4] for box in boxes]
                scores = [box[4] for box in boxes]
                cls_ids = [int(box[5]) for box in boxes]
                frame = vis(frame, box_coords, scores, cls_ids, 
                            conf=0.5, class_names=CLASSES)
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
                            self.command.servo_control('plastic') 

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
                cv2.imshow("Webcam Inference", result_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                self.command.servo_reset()
                self.command.send_message('R')
                break
        cap.release()
        cv2.destroyAllWindows()
