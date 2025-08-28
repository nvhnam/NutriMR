import json
import socket
import struct
import numpy as np
import cv2
from ultralytics import YOLO
import torch
import time
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Choose protocol to use (UDP or TCP).")
    parser.add_argument("--protocol", type=str, default="udp", choices=["udp", "tcp"], help="Protocol to use (udp or tcp)")
    return parser.parse_args()


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
UNITY_IP = "192.168.1.7"  
UNITY_PORT = 5011      
CHUNK_SIZE = 1200  

def load_model():
    print("Loading YOLO model...")
    model = YOLO("model/yolov10/YOLOv10b_VietFood67_SGD_new_bigger.pt")   
    # model = YOLO("yolov8n.pt")   
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)
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
                # "xyxy": {
                #     "x1": xyxy[0],  # top-left-x
                #     "y1": xyxy[1],  # top-left-y
                #     "x2": xyxy[2],  # bottom-right-x
                #     "y2": xyxy[3],  # bottom-right-y
                # },
                "confidence": float(conf)
            })
    return detections

# This is for TCP
def recv_all(sock, length):
    """Receive exactly `length` bytes from a TCP socket"""
    buf = b''
    while len(buf) < length:
        data = sock.recv(length - len(buf))
        if not data:
            return None  # connection closed
        buf += data
    return buf

def receive_image(sock, protocol):
    if (protocol == "udp"):
        chunks = {}
        total_chunks = None
        start_time = time.time()

        while True:
            try:
                data, _ = sock.recvfrom(CHUNK_SIZE + 2)
                chunk_index = data[0]
                total_chunks = data[1]
                chunk_data = data[2:]

                # Store the chunk
                if chunk_index not in chunks:
                    chunks[chunk_index] = chunk_data
                else:
                    print(f"Duplicate chunk received: {chunk_index}")

                # Check if all chunks are received
                if len(chunks) == total_chunks:
                    break

                # Timeout to avoid infinite waiting
                if time.time() - start_time > 5:  # 5 seconds timeout
                    raise TimeoutError("Timeout while waiting for all chunks")
                    break

            except Exception as e:
                # print(f"Error receiving chunk: {e}")
                return None

        # Reassemble the image data
        try:
            image_data = b"".join(chunks[i] for i in range(total_chunks))
        except KeyError as e:
            print(f"Missing chunk: {e}")
            return None

        # Decode the image
        arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    elif (protocol == "tcp"):
        length_data = recv_all(sock, 4)
        if not length_data:
            return None
        frame_length = struct.unpack('<I', length_data)[0]

        # 2. Read JPEG data
        frame_data = recv_all(sock, frame_length)
        if frame_data is None:
            return None

        # 3. Decode JPEG into numpy array (OpenCV BGR image)
        arr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame

def init_frame_network(protocol):
    if (protocol == "udp"):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((LISTEN_IP, LISTEN_PORT))
        sock.settimeout(0.01)
    elif (protocol == "tcp"):
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.bind((LISTEN_IP, LISTEN_PORT))
        server_sock.listen(1)
        print(f"Listening for TCP connection on {LISTEN_IP}:{LISTEN_PORT}...")
        sock, addr = server_sock.accept()
        print(f"Accepted TCP connection from {addr}")

    print(f"Listening for frames on {LISTEN_IP}:{LISTEN_PORT}...")
    return sock

def init_label_network(protocol):
    # Need to send label to the UNITY IP
    if protocol == "udp":
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Allow address reuse
    elif protocol == "tcp":
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Allow address reuse
        try:
            sock.connect((UNITY_IP, UNITY_PORT))  # Connect to Unity server
        except Exception as e:
            print(f"Error connecting to Unity server: {e}")
            sock.close()
            return None
    return sock

def send_label(protocol, sock, message):
    if (protocol == "udp"):
        # Send back to Unity
        sock.sendto(message, (UNITY_IP, UNITY_PORT))
    elif protocol == "tcp":
        # Prefix message with its length (4 bytes, little-endian)
        msg_len = struct.pack('<I', len(message))
        sock.sendall(msg_len + message)
    print('Sent detections via:', protocol)

def main(protocol):
    print('Protocol:', protocol)
    model = load_model()
    sock = init_frame_network(protocol)
    sockUDP = init_label_network("udp")
    try:
        prev_time = time.time()
        while True:
            try:
                # data, addr = sock.recvfrom(200_000)
                # arr = np.frombuffer(data, np.uint8)
                # frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                frame = receive_image(sock, protocol)
                if frame is None:
                    continue

                # Calculate FPS
                current_time = time.time()
                fps = 1 / (current_time - prev_time)
                prev_time = current_time

                # print('Input resolution is:', frame.shape[:2])

                # Run inference
                results = do_inference(0.7, frame, model)

                # Convert results to structured detections
                detections = build_detections(results)

                # Encode as JSON
                message = json.dumps(detections).encode("utf-8")

                
                if sockUDP is None:
                    print("Failed to initialize label network. Skipping this frame.")

                # Send to Unity
                send_label("udp", sockUDP, message)
                # (Optional) Show annotated frame for debugging
                annotated = results[0].plot()
                # Overlay FPS on the video feed
                cv2.putText(annotated, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Received Frame", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

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
    args = parse_args()
    protocol = args.protocol
    main(protocol)
