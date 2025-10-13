import torch
from ultralytics import YOLO
import networkmanager
import cv2
import time
import json
import socket

class ModelManager:
    def __init__(self, protocol, no_split):
        self.model = None
        self.load_model()
        self.networkManager = networkmanager.NetworkManager(protocol=protocol, UNITY_IP="10.0.10.129", UNITY_PORT=5014, LISTEN_PORT=5010, 
        no_split=no_split)
        self.networkManager.init_frame_network()
    def load_model(self):
        print("Loading YOLO model...")
        self.model = YOLO("model/yolov10/YOLOv10b_VietFood67_SGD_new_bigger.pt")   
        # self.model = YOLO("yolov8n.pt")   
        device = "mps" if torch.backends.mps.is_available() else \
                "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        self.model.to(device)
    def do_inference(self, conf, image):
        results = self.model.predict(image, conf=conf, imgsz=640, verbose=False)
        return results
    def build_detections(self, results):
        detections = []
        for result in results[0]: # Remember to use results[0] for YOLOv10 model and results for YOLOv8 model
            boxes = result.boxes
            for xywh, xyxy, cls, conf in zip(
                boxes.xywh.tolist(),
                boxes.xyxy.tolist(),
                boxes.cls.int().tolist(),
                boxes.conf.tolist()
            ):
                class_name = result.names[cls]
                if class_name == "Con nguoi" or class_name == "Con nguoi (Human)" or class_name == "Human":
                    continue  # Skip human detections
                detections.append({
                    "class": result.names[cls],
                    "bbox": {
                        "cx": xywh[0],  # center-x
                        "cy": xywh[1],  # center-y
                        "w": xywh[2],   # width
                        "h": xywh[3],   # height
                    },
                    # "xyxy": {
                    #     "x1": xyxy[0],  # top-left-x
                    #     "y1": xyxy[1],  # top-left-y
                    #     "x2": xyxy[2],  # bottom-right-x
                    #     "y2": xyxy[3],  # bottom-right-y
                    # },
                    "confidence": float(conf)
                })

        # select the detection with the highest confidence for each class
        if detections:
            unique_classes = set(d['class'] for d in detections)
            filtered_detections = []
            for cls in unique_classes:
                class_detections = [d for d in detections if d['class'] == cls]
                best_detection = max(class_detections, key=lambda x: x['confidence'])
                filtered_detections.append(best_detection)
            detections = filtered_detections
        return detections
    