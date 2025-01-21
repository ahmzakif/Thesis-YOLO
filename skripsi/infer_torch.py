import time
import glob
from pathlib import Path

import cv2
import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

from .classes import CLASSES
from .speed import SpeedTracker
from .visualize import vis


class InferTorch():
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.init_model()

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

    def run_image(self, img_path, enable_vis=False, write_output=False):
        img = cv2.imread(img_path)
        inference_time = end - start

        # Preprocess image
        img_tensor, ratio = self.preprocess(img)
        start = time.time()

        # Run inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
        end = time.time()

        # Postprocess results
        boxes = self.postprocess(outputs, ratio)

        if enable_vis:
            result_img = self.visualize(img, boxes)
            plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            plt.show()

            print(f"Inference Time: {inference_time * 1000:.2f} ms")
            print(f"FPS: {1 / inference_time:.2f}")
            cv2.imwrite('./outputs/prediction.jpg', result_img)

        if write_output:
            output_path = Path('./outputs')
            output_path.mkdir(exist_ok=True, parents=True)
            output_txt = output_path / (Path(img_path).stem + ".txt")

            with open(output_txt, 'w') as f:
                for box in boxes:
                    f.write(" ".join(map(str, box)) + "\n")

        return boxes
    
    def run_video(self, video_path, enable_vis=False):
        cap = cv2.VideoCapture(video_path)
        imageWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        imageHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('./outputs/output.avi', fourcc, fps, (imageWidth, imageHeight))

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break
            
            if ret:
                # Preprocess frame
                img_tensor, ratio = self.preprocess(frame)
                start_time = time.time()

                # Run inference
                with torch.no_grad():
                    outputs = self.model(img_tensor)
                
                # Postprocess results
                boxes = self.postprocess(outputs, ratio)

                end_time = time.time()
                inference_time = end_time - start_time

                print(f"Inference Time: {inference_time * 1000:.2f} ms")
                print(f"FPS: {1 / inference_time:.2f}")

                if enable_vis:
                    result_img = self.visualize(frame, boxes, conf_thr=0.5)
                    resized_img = cv2.resize(result_img, (imageWidth, imageHeight))

                    cv2.imshow('Video Inference', resized_img)
                    
                    out.write(result_img)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()
    
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
                # Preprocess frame
                img_tensor, ratio = self.preprocess(frame)
                start_time  = time.time()

                # Run inference
                with torch.no_grad():
                    outputs = self.model(img_tensor, verbose=False)
                
                # Postprocess results
                boxes = self.postprocess(outputs, ratio) 

                end_time = time.time()
                inference_time = end_time - start_time
                
                if enable_vis:
                    result_img = self.visualize(frame, boxes, conf_thr=0.5)

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
            self.run(img_path, enable_vis=False, write_output=True)
