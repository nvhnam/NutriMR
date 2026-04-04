"""
test_mac_discovery.py — simulates the Mac side of IP discovery.

Run this in one terminal, and test_hololens_sim.py in another.
Both should see each other's IPs within a few seconds.

Ports (must match networkmanager.py and NetworkDiscovery.cs):
  5010 — HoloLens broadcasts "HOLOLENS:{ip}" → Mac listens here
  5011 — Mac broadcasts "MAC:{ip}"            → HoloLens listens
"""

import socket
import threading
import time

HOLOLENS_BROADCAST_PORT = 5010   # we listen here
MAC_BROADCAST_PORT      = 5011   # we send here

FAST_INTERVAL = 2    # seconds, first 60 s
FAST_DURATION = 60   # seconds
SLOW_INTERVAL = 10   # seconds

hololens_ip = None
running     = True

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip

def broadcast_mac_ip(local_ip):
    message    = f"MAC:{local_ip}".encode()
    start_time = time.time()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        while running:
            s.sendto(message, ("<broadcast>", MAC_BROADCAST_PORT))
            elapsed  = time.time() - start_time
            interval = FAST_INTERVAL if elapsed < FAST_DURATION else SLOW_INTERVAL
            print(f"[MacSim]  Broadcast MAC:{local_ip}  (interval={interval}s)")
            time.sleep(interval)

def listen_for_hololens():
    global hololens_ip
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        s.bind(("0.0.0.0", HOLOLENS_BROADCAST_PORT))
        s.settimeout(1.0)
        while running:
            try:
                data, _ = s.recvfrom(256)
                try:
                    msg = data.decode().strip()
                except UnicodeDecodeError:
                    continue  # binary packet (e.g. JPEG frame) on port 5010 — ignore
                if msg.startswith("HOLOLENS:"):
                    ip = msg[len("HOLOLENS:"):]
                    if ip != hololens_ip:
                        hololens_ip = ip
                        print(f"[MacSim]  ✓ HoloLens IP discovered: {hololens_ip}")
            except socket.timeout:
                continue

if __name__ == "__main__":
    local_ip = get_local_ip()
    print(f"[MacSim]  Mac IP: {local_ip}")
    print("[MacSim]  Fast phase: 2 s interval for 60 s, then 10 s")
    print("-" * 50)

    threading.Thread(target=broadcast_mac_ip, args=(local_ip,), daemon=True).start()
    threading.Thread(target=listen_for_hololens, daemon=True).start()

    try:
        while True:
            time.sleep(5)
            print(f"[MacSim]  Status — HoloLens IP: {hololens_ip or 'not yet discovered'}")
    except KeyboardInterrupt:
        running = False
        print("\n[MacSim]  Stopped.")
