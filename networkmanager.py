from datetime import datetime
import struct
import socket
import numpy as np
import cv2
import time

from ultralytics import data
class NetworkManager:
    LISTEN_IP = "0.0.0.0"           
    CHUNK_SIZE = 1200  
    def __init__(self, protocol, UNITY_IP, UNITY_PORT, LISTEN_PORT, no_split):
        self.protocol = protocol
        self.UNITY_IP = UNITY_IP
        self.UNITY_PORT = UNITY_PORT
        self.LISTEN_PORT = LISTEN_PORT
        self.sock_label = self.init_label_network("udp")
        self.no_split = no_split
        self.call_number = 0

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
            self.sock_frame = sock
            return

        elif self.protocol == "udp":
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((self.LISTEN_IP, self.LISTEN_PORT))
            sock.settimeout(0.01)
            self.sock_frame = sock
            return

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
            if (self.no_split == False):
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
            else:
                try:
                    data, _ = self.sock_frame.recvfrom(200_000)
                    arr = np.frombuffer(data, np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    return frame
                except socket.timeout:
                    pass
                except Exception as e:
                    print("Error:", e)
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
    def change_unity_ip(self, new_ip):
        self.UNITY_IP = new_ip
        self.refresh()
    
    def receive_file(self, port):
        print(f'call number: {self.call_number}')
        self.call_number += 1
        if self.protocol == "udp":
            # tcp only
            pass
        elif self.protocol == "tcp":
            HOST = "0.0.0.0"
            BUFFER_SIZE = 4096
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((HOST, port))
                s.listen(1)
                print('Waiting for connection...')
                conn, addr = s.accept()
                with conn:
                    print('Connected by', addr)
                    # get current date and time
                    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    file_name = f"received_file_{time}.csv"
                    with open(file_name, "wb") as f:
                        while True:
                            data = conn.recv(BUFFER_SIZE)
                            if not data:
                                break
                            f.write(data)
                    print(f"File received successfully at {time}")
