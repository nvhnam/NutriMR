import socket
import threading
import time

HOLOLENS_BROADCAST_PORT = 5010   # HoloLens → Mac
MAC_BROADCAST_PORT = 5011        # Mac → HoloLens
BROADCAST_INTERVAL = 2

hololens_ip = None
running = True

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip

def broadcast_mac_ip(local_ip):
    message = f"MAC:{local_ip}".encode()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        while running:
            s.sendto(message, ("<broadcast>", MAC_BROADCAST_PORT))
            time.sleep(BROADCAST_INTERVAL)

def listen_for_hololens():
    global hololens_ip
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", HOLOLENS_BROADCAST_PORT))
        s.settimeout(1.0)
        while running:
            try:
                data, _ = s.recvfrom(256)
                msg = data.decode().strip()
                if msg.startswith("HOLOLENS:"):
                    ip = msg[len("HOLOLENS:"):]
                    if ip != hololens_ip:
                        hololens_ip = ip
                        print(f"[Discovery] HoloLens IP found: {hololens_ip}")
            except socket.timeout:
                continue

local_ip = get_local_ip()
print(f"[Discovery] Mac local IP: {local_ip}")

threading.Thread(target=broadcast_mac_ip, args=(local_ip,), daemon=True).start()
threading.Thread(target=listen_for_hololens, daemon=True).start()

for _ in range(20):
    print(f"HoloLens IP: {hololens_ip}")
    time.sleep(1)

running = False
