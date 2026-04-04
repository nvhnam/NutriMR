from datetime import datetime
import struct
import socket
import numpy as np
import cv2
import time
import threading

from ultralytics import data

# Ports for IP discovery handshake
_HOLOLENS_BROADCAST_PORT = 5010   # HoloLens → Mac
_MAC_BROADCAST_PORT = 5011        # Mac → HoloLens

# Broadcast timing:
#   0–60 s : every 2 s  (fast phase — acquire IP quickly)
#   60 s + : every 10 s (slow verify phase)
_FAST_INTERVAL = 2    # seconds
_FAST_DURATION = 60   # seconds
_SLOW_INTERVAL = 10   # seconds

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

        self._hololens_ip = None
        self._discovery_running = False
        self._discovery_threads = []

    # ------------------------------------------------------------------ #
    #  IP Discovery                                                        #
    # ------------------------------------------------------------------ #

    @property
    def hololens_ip(self):
        return self._hololens_ip

    @staticmethod
    def _get_local_ip():
        """Return the Mac's primary local (LAN) IP address."""
        try:
            # Connect to a public address without sending data — just to resolve route
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "0.0.0.0"

    def start_ip_discovery(self):
        """
        Start two background threads:
          1. Broadcast this Mac's IP on port 5011 every 2 s
          2. Listen on port 5010 for the HoloLens IP
        Discovered HoloLens IP is stored in self.hololens_ip and also
        applied automatically via change_unity_ip().
        """
        if self._discovery_running:
            return
        self._discovery_running = True

        t_broadcast = threading.Thread(target=self._broadcast_mac_ip, daemon=True)
        t_listen = threading.Thread(target=self._listen_for_hololens_ip, daemon=True)

        self._discovery_threads = [t_broadcast, t_listen]
        t_broadcast.start()
        t_listen.start()
        print(f"[Discovery] Started — Mac IP: {self._get_local_ip()}")

    def stop_ip_discovery(self):
        self._discovery_running = False

    def _broadcast_mac_ip(self):
        local_ip = self._get_local_ip()
        message = f"MAC:{local_ip}".encode()
        start_time = time.time()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            while self._discovery_running:
                try:
                    s.sendto(message, ("<broadcast>", _MAC_BROADCAST_PORT))
                except Exception as e:
                    print(f"[Discovery] Broadcast error: {e}")
                elapsed = time.time() - start_time
                interval = _FAST_INTERVAL if elapsed < _FAST_DURATION else _SLOW_INTERVAL
                time.sleep(interval)

    def _listen_for_hololens_ip(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", _HOLOLENS_BROADCAST_PORT))
            s.settimeout(1.0)
            while self._discovery_running:
                try:
                    data, _ = s.recvfrom(256)
                    try:
                        message = data.decode().strip()
                    except UnicodeDecodeError:
                        continue  # binary packet (e.g. JPEG frame) on port 5010 — ignore
                    if message.startswith("HOLOLENS:"):
                        ip = message[len("HOLOLENS:"):]
                        if ip != self._hololens_ip:
                            self._hololens_ip = ip
                            print(f"[Discovery] HoloLens IP: {ip}")
                            self.change_unity_ip(ip)
                except socket.timeout:
                    continue
                except Exception as e:
                    if self._discovery_running:
                        print(f"[Discovery] Listen error: {e}")

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
