import json
import socket
import numpy as np
import cv2
from ultralytics import YOLO
import torch


# def load_model():
#     print("breakpoint 1")
#     model = YOLO("yolov8n.pt")
#     device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")
#     return model

# # def preprocess(image):
# #     print("breakpoint 2")
# #     img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# #     img = cv2.resize(img, (640, 640))
# #     img = img.astype(np.float32) / 255.0
# #     img = img.astype(np.float16)
# #     img = np.transpose(img, (2, 0, 1))
# #     img = np.expand_dims(img, axis=0)
# #     return img


# def do_inference(conf, image, model):
#     print("breakpoint 3")
#     res = model.predict(image, conf=conf, imgsz=640)
#     return res


# # # IP and port to listen on
# # LISTEN_IP = "0.0.0.0"   # Listen on all interfaces
# # LISTEN_PORT = 5010      # Must match sender’s port



# model = load_model()  # Load the YOLO model
# # model.to(device)  # Move the model to the specified device

# # def main():
# #     # Create and bind a UDP socket
# #     sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# #     sock.bind((LISTEN_IP, LISTEN_PORT))
# #     sock.settimeout(0.01)  # short timeout so we can update the window even if a packet is late

# #     # Prepare the window
# #     cv2.namedWindow("Received Frame", cv2.WINDOW_NORMAL)
# #     cv2.startWindowThread()

# #     frame = None
# #     try:
# #         while True:
# #             # 1. Try to receive one JPEG packet (non-blocking-ish)
# #             try:
# #                 data, _ = sock.recvfrom(200_000)
# #                 arr = np.frombuffer(data, np.uint8)
# #                 decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
# #                 if decoded is not None:
# #                     frame = decoded
# #             except socket.timeout:
# #                 pass

# #             # 2. Always display the latest frame
# #             if frame is not None:
# #                 print(type(frame))
# #                 # results = do_inference(0.1, preprocess(frame), model)                # Run inference
# #                 # print(results[0].boxes)               # Print detection results
# #                 # annotated_frame = results[0].plot()   # Draw detections
# #                 # cv2.imshow("Received Frame", annotated_frame)
                
# #                 # cv2.imshow("Received Frame", frame)

# #             # 3. Pump the GUI event loop (no key checks)
# #             # if cv2.waitKey(1) & 0xFF == ord('q'):
# #             #     break

# #     except KeyboardInterrupt:
# #         pass
# #     finally:
# #         sock.close()
# #         cv2.destroyAllWindows()

# def main():
#     print("Starting inference...")
#     image = cv2.imread("D:\\NutriMR\\NutriMR\\f35.png")
#     # print(image)
#     if image is None:
#         raise FileNotFoundError("Image not found! Check your path again.")
#     # processed_img = preprocess(image)
#     results = do_inference(0.1, image, model)
#     print(f"result: {results}")
#     print(type(results))
#     print(results[0].boxes)  # Print detection results
#     annotated_frame = results[0].plot()  # Draw detections
#     cv2.imshow("Received Frame", annotated_frame)
#     cv2.waitKey(0)

# if __name__ == "__main__":
#     main()




### -------------------------------------------------------- ###

# --------------------- Nam Code ------------------------------#

LISTEN_IP = "0.0.0.0"     
LISTEN_PORT = 5010      
UNITY_IP = "127.0.0.1"  
UNITY_PORT = 5011      
CHUNK_SIZE = 4096  

def load_model():
    print("Loading YOLO model...")
    model = YOLO("YOLOv10b_VietFood67_SGD_new_bigger.pt")   
    # model = YOLO("yolov8n.pt")   
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return model

def do_inference(conf, image, model):
    results = model.predict(image, conf=conf, imgsz=640, verbose=False)
    return results

def build_detections(results):
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
                "xyxy": {
                    "x1": xyxy[0],  # top-left-x
                    "y1": xyxy[1],  # top-left-y
                    "x2": xyxy[2],  # bottom-right-x
                    "y2": xyxy[3],  # bottom-right-y
                },
                "confidence": float(conf)
            })
    return detections

def receive_image(sock):
    num_chunks_data, _ = sock.recvfrom(1024)
    num_chunks = int(num_chunks_data.decode())
    img_bytes = b""
    for _ in range(num_chunks):
        chunk, _ = sock.recvfrom(CHUNK_SIZE)
        img_bytes += chunk

    arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame

def main():
    model = load_model()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((LISTEN_IP, LISTEN_PORT))
    sock.settimeout(0.01)

    print(f"Listening for frames on {LISTEN_IP}:{LISTEN_PORT}...")

    try:
        while True:
            try:
                # data, addr = sock.recvfrom(200_000)
                # arr = np.frombuffer(data, np.uint8)
                # frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                frame = receive_image(sock)
                
                if frame is None:
                    continue

                # Run inference
                results = do_inference(0.3, frame, model)

                # Convert results to structured detections
                detections = build_detections(results)

                # Encode as JSON
                message = json.dumps(detections).encode("utf-8")

                # Send back to Unity
                sock.sendto(message, (UNITY_IP, UNITY_PORT))

                # (Optional) Show annotated frame for debugging
                # annotated = results[0].plot()
                # cv2.imshow("Received Frame", annotated)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

            except socket.timeout:
                pass
            except Exception as e:
                print("Error:", e)

    except KeyboardInterrupt:
        print("Shutting down...")

    finally:
        sock.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
