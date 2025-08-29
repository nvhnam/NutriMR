import struct
import socket
import numpy as np
import cv2
import time
class NetworkManager:
    LISTEN_IP = "0.0.0.0"     
    LISTEN_PORT = 5010      
    UNITY_IP = "192.168.1.7"  
    UNITY_PORT = 5011      
    CHUNK_SIZE = 1200  
    def __init__(self, protocol):
        self.protocol = protocol
        self.sock_label = self.init_label_network("udp")
        self.sock_frame = self.init_frame_network()
    # This is for TCP
    def recv_all(self, length):
        """Receive exactly `length` bytes from a TCP socket"""
        buf = b''
        while len(buf) < length:
            data = self.sock_frame.recv(length - len(buf))
            if not data:
                return None  # connection closed
            buf += data
        return buf
    def init_frame_network(self):
        if self.protocol == "tcp":
            self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_sock.bind((self.LISTEN_IP, self.LISTEN_PORT))
            self.server_sock.listen(1)
            print(f"Listening for TCP connection on {self.LISTEN_IP}:{self.LISTEN_PORT}...")
            sock, addr = self.server_sock.accept()
            print(f"Accepted TCP connection from {addr}")
            return sock
        elif self.protocol == "udp":
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((self.LISTEN_IP, self.LISTEN_PORT))
            sock.settimeout(0.01)
            return sock

    def init_label_network(self, protocol):
        if protocol == "udp":
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Allow address reuse
        elif protocol == "tcp":
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Allow address reuse
            try:
                sock.connect((self.UNITY_IP, self.UNITY_PORT))  # Connect to Unity server
            except Exception as e:
                print(f"Error connecting to Unity server: {e}")
                sock.close()
                return None
        return sock
    def send_label(self, protocol, message):
        if (protocol == "udp"):
            self.sock_label.sendto(message, (self.UNITY_IP, self.UNITY_PORT))
        elif protocol == "tcp":
            # Prefix message with its length (4 bytes, little-endian)
            msg_len = struct.pack('<I', len(message))
            self.sock_label.sendall(msg_len + message)
        # print('Sent detections via:', protocol)
    def receive_image(self):
        if (self.protocol == "udp"):
            chunks = {}
            total_chunks = None
            start_time = time.time()

            while True:
                try:
                    data, _ = self.sock_frame.recvfrom(self.CHUNK_SIZE + 2)
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
        elif (self.protocol == "tcp"):
            length_data = self.recv_all(4)
            if not length_data:
                return None
            frame_length = struct.unpack('<I', length_data)[0]

            # 2. Read JPEG data
            frame_data = self.recv_all(frame_length)
            if frame_data is None:
                return None

            # 3. Decode JPEG into numpy array (OpenCV BGR image)
            arr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
    def self_destruct(self):
        self.sock_label.close()
        self.sock_frame.close()
    def refresh(self):
        if self.sock_frame:
            try:
                self.sock_frame.close()
            except OSError:
                pass
            self.sock_frame = None

        if hasattr(self, "server_sock") and self.server_sock:
            try:
                self.server_sock.close()
            except OSError:
                pass
            self.server_sock = None

        self.sock_frame = self.init_frame_network()

