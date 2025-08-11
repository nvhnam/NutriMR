import socket
import numpy as np
import cv2
from ultralytics import YOLO
import torch
def load_model():
    print("breakpoint 1")
    model = YOLO("yolov8n.pt")
    # # Check if MPS is available and set the device
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return model

model = load_model()  # Load the YOLO model