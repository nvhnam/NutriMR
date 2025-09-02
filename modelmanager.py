import torch
from ultralytics import YOLO
import networkmanager
import cv2
import time
import json
import socket

class ModelManager:
    def __init__(self, protocol):
        self.model = None
        self.load_model()
        self.networkManager = networkmanager.NetworkManager(protocol=protocol, UNITY_IP="192.168.1.7", UNITY_PORT=5011, LISTEN_PORT=5010
        , no_split=False)
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
        return detections
    