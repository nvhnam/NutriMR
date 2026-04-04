"""
test_hololens_sim.py — simulates the HoloLens side of IP discovery.

Run this in one terminal, and test_mac_discovery.py in another.
Both should see each other's IPs within a few seconds.

Ports (must match networkmanager.py and NetworkDiscovery.cs):
  5010 — HoloLens broadcasts "HOLOLENS:{ip}" → Mac listens
  5011 — Mac broadcasts "MAC:{ip}"            → HoloLens listens
"""

import socket
import threading
import time

HOLOLENS_BROADCAST_PORT = 5010   # we send here
MAC_LISTEN_PORT          = 5011   # we listen here

FAST_INTERVAL = 2    # seconds, first 60 s
FAST_DURATION = 60   # seconds
SLOW_INTERVAL = 10   # seconds

mac_ip  = None
running = True

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip

def broadcast_hololens_ip(local_ip):
    message    = f"HOLOLENS:{local_ip}".encode()
    start_time = time.time()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        while running:
            s.sendto(message, ("<broadcast>", HOLOLENS_BROADCAST_PORT))
            elapsed  = time.time() - start_time
            interval = FAST_INTERVAL if elapsed < FAST_DURATION else SLOW_INTERVAL
            print(f"[HoloSim] Broadcast HOLOLENS:{local_ip}  (interval={interval}s)")
            time.sleep(interval)

def listen_for_mac():
    global mac_ip
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", MAC_LISTEN_PORT))
        s.settimeout(1.0)
        while running:
            try:
                data, _ = s.recvfrom(256)
                msg = data.decode().strip()
                if msg.startswith("MAC:"):
                    ip = msg[len("MAC:"):]
                    if ip != mac_ip:
                        mac_ip = ip
                        print(f"[HoloSim] ✓ Mac IP discovered: {mac_ip}")
            except socket.timeout:
                continue

if __name__ == "__main__":
    local_ip = get_local_ip()
    print(f"[HoloSim] Simulated HoloLens IP: {local_ip}")
    print("[HoloSim] Fast phase: 2 s interval for 60 s, then 10 s")
    print("-" * 50)

    threading.Thread(target=broadcast_hololens_ip, args=(local_ip,), daemon=True).start()
    threading.Thread(target=listen_for_mac, daemon=True).start()

    try:
        while True:
            time.sleep(5)
            print(f"[HoloSim] Status — Mac IP: {mac_ip or 'not yet discovered'}")
    except KeyboardInterrupt:
        running = False
        print("\n[HoloSim] Stopped.")
