import math
import socket
import numpy as np
import cv2
from ultralytics import YOLO
import torch
# def load_model():
#     print("breakpoint 1")
#     model = YOLO("yolov8n.pt")
#     # # Check if MPS is available and set the device
#     device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")
#     return model

# model = load_model()  # Load the YOLO model




## --------------------------------------------- ##
## ---------------- Nam code ------------------- ##

import socket
import cv2
import json

SERVER_IP = "127.0.0.1"   
SERVER_PORT = 5010        
CLIENT_PORT = 5011        
CHUNK_SIZE = 4096

def main():
    img = cv2.imread("D:\\NutriMR\\NutriMR\\noodle.webp")
    # Please change the img directory path to match with your Mac machine
    if img is None:
        raise FileNotFoundError("Could not load image. Check the path to noodle.jpg")

    img = cv2.resize(img, (640, 640))
    success, encoded = cv2.imencode(".jpg", img)
    if not success:
        raise RuntimeError("Failed to encode image")

    data = encoded.tobytes()
    data_len = len(data)
    print(f"Encoded image size: {data_len} bytes")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", CLIENT_PORT)) 

    num_chunks = math.ceil(data_len / CHUNK_SIZE)
    sock.sendto(str(num_chunks).encode(), (SERVER_IP, SERVER_PORT))
    print(f"Sending {num_chunks} chunks...")

    for i in range(num_chunks):
        start = i * CHUNK_SIZE
        end = start + CHUNK_SIZE
        chunk = data[start:end]
        sock.sendto(chunk, (SERVER_IP, SERVER_PORT))

    sock.settimeout(20.0)
    try:
        response, _ = sock.recvfrom(200_000)
        detections = json.loads(response.decode("utf-8"))
        print("Received detections:")
        print(json.dumps(detections, indent=2))
    except socket.timeout:
        print("No response from server")
    finally:
        sock.close()

if __name__ == "__main__":
    main()
