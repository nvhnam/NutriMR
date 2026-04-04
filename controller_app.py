"""
controller_app.py — Mac-side HoloLens controller.

Layout:
  Left panel (top)    — Tracking controls (eye + head)
  Left panel (bottom) — Audio recording
  Right panel         — Live HoloLens feed + YOLO inference

Requirements:
  pip install sounddevice soundfile Pillow opencv-python
"""

import os
import json
import time
import threading
import socket
from datetime import datetime

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

try:
    import sounddevice as sd
    import soundfile as sf
    _AUDIO_OK = True
except ImportError:
    _AUDIO_OK = False

import torch
from ultralytics import YOLO

from networkmanager import NetworkManager

_MODEL_PATH = "model/yolov10/YOLOv10b_VietFood67_SGD_new_bigger.pt"
_SKIP_CLASSES = {"Con nguoi", "Con nguoi (Human)", "Human"}


def _load_yolo():
    model = YOLO(_MODEL_PATH)
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"[App] YOLO loaded on {device}")
    return model


def _build_detections(results) -> list:
    detections = []
    for result in results[0]:
        boxes = result.boxes
        for xywh, cls, conf in zip(
            boxes.xywh.tolist(),
            boxes.cls.int().tolist(),
            boxes.conf.tolist(),
        ):
            class_name = result.names[cls]
            if class_name in _SKIP_CLASSES:
                continue
            detections.append({
                "class": class_name,
                "bbox": {"cx": xywh[0], "cy": xywh[1], "w": xywh[2], "h": xywh[3]},
                "confidence": float(conf),
            })
    # Keep only highest-confidence detection per class
    if detections:
        best = {}
        for d in detections:
            if d["class"] not in best or d["confidence"] > best[d["class"]]["confidence"]:
                best[d["class"]] = d
        detections = list(best.values())
    return detections

# ── Ports ────────────────────────────────────────────────────────────── #
CMD_PORT       = 5015   # Mac → HoloLens  commands (UDP)
EYE_FILE_PORT  = 5012   # HoloLens → Mac  eye CSV  (TCP)
HEAD_FILE_PORT = 5013   # HoloLens → Mac  head CSV (TCP)
FRAME_PORT     = 5010   # HoloLens → Mac  frames   (UDP)
LABEL_PORT     = 5014   # Mac → HoloLens  labels   (UDP)

DATA_DIR = "participant_data"

# ── Palette (dark theme) ─────────────────────────────────────────────── #
BG      = "#1e1e1e"
PANEL   = "#252526"
BTN     = "#3a3d41"
BTN_GO  = "#0d6b0d"
BTN_REC = "#9b1a1a"
FG      = "#d4d4d4"
ACCENT  = "#4ec9b0"
WARN    = "#ce9178"
FONT    = "Helvetica"


def _lighten(hex_color: str, amount: int = 30) -> str:
    """Return a slightly lighter shade of a hex color string."""
    r = min(255, int(hex_color[1:3], 16) + amount)
    g = min(255, int(hex_color[3:5], 16) + amount)
    b = min(255, int(hex_color[5:7], 16) + amount)
    return f"#{r:02x}{g:02x}{b:02x}"


class ControllerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("HoloLens Controller")
        self.root.configure(bg=BG)
        self.root.geometry("1350x800")
        self.root.minsize(1000, 620)

        # Audio state
        self._recording    = False
        self._audio_chunks: list = []
        self._audio_start: datetime | None = None
        self._audio_stream = None
        self._sample_rate  = 44100

        # UDP socket for sending commands to HoloLens
        self._cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # NetworkManager handles frame rx, label tx, and IP discovery
        self._nm = NetworkManager(
            "udp", "0.0.0.0", LABEL_PORT, FRAME_PORT, no_split=False)
        self._nm.start_ip_discovery()

        # YOLO model (optional — app still works for control without it)
        try:
            self._model = _load_yolo()
        except Exception as e:
            print(f"[App] YOLO unavailable: {e}")
            self._model = None

        os.makedirs(DATA_DIR, exist_ok=True)

        self._build_ui()

        # Init UDP frame socket — optional, app works without it
        try:
            self._nm.init_frame_network()
            threading.Thread(target=self._frame_loop, daemon=True).start()
        except OSError as e:
            print(f"[App] Frame socket unavailable ({e}) — live feed disabled")

        # Poll HoloLens connection status every 2 s
        self._poll_connection()

    # ─────────────────────────────────────────────────────────────────── #
    # UI
    # ─────────────────────────────────────────────────────────────────── #

    def _build_ui(self):
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left  = tk.Frame(self.root, bg=PANEL, width=300)
        right = tk.Frame(self.root, bg="#000000")
        left.grid(row=0, column=0, sticky="nsew")
        right.grid(row=0, column=1, sticky="nsew")
        left.grid_propagate(False)

        self._build_left(left)
        self._build_right(right)

    def _build_left(self, p):
        # ── Participant name ──────────────────────────────────────────── #
        sec = self._section(p, "Participant")
        self._name_var = tk.StringVar(value="P01")
        tk.Entry(
            sec, textvariable=self._name_var,
            bg="#3c3c3c", fg="white", insertbackground="white",
            relief="flat", font=(FONT, 13), bd=4,
        ).pack(fill="x", padx=10, pady=8)

        # ── TOP LEFT — Eye & head tracking ───────────────────────────── #
        sec = self._section(p, "Tracking Controls")

        self._eye_var  = tk.StringVar(value="Eye tracking: idle")
        self._head_var = tk.StringVar(value="Head tracking: idle")

        self._btn(sec, "▶  Start Eye Tracking",  self._start_eye,  BTN_GO)
        self._btn(sec, "■  Stop Eye Tracking",   self._stop_eye)
        tk.Label(sec, textvariable=self._eye_var, bg=PANEL, fg=ACCENT,
                 font=(FONT, 10)).pack(anchor="w", padx=14, pady=(0, 8))

        self._btn(sec, "▶  Start Head Tracking", self._start_head, BTN_GO)
        self._btn(sec, "■  Stop Head Tracking",  self._stop_head)
        tk.Label(sec, textvariable=self._head_var, bg=PANEL, fg=ACCENT,
                 font=(FONT, 10)).pack(anchor="w", padx=14, pady=(0, 6))

        self._btn(sec, "⬇  Receive & Save Tracking Data",
                  self._receive_files, "#0a5a8a")

        # ── BOTTOM LEFT — Audio recording ─────────────────────────────── #
        sec = self._section(p, "Audio Recording")
        self._audio_var = tk.StringVar(
            value="Ready" if _AUDIO_OK else "sounddevice not installed")

        self._btn(sec, "⏺  Start Recording", self._start_audio, BTN_REC)
        self._btn(sec, "■  Stop & Save",     self._stop_audio)
        tk.Label(
            sec, textvariable=self._audio_var, bg=PANEL, fg=WARN,
            font=(FONT, 10), wraplength=260, justify="left",
        ).pack(anchor="w", padx=14, pady=6)

        # ── Connection status ─────────────────────────────────────────── #
        sec = self._section(p, "Connection")
        self._conn_var = tk.StringVar(value="Searching for HoloLens…")
        tk.Label(
            sec, textvariable=self._conn_var, bg=PANEL, fg=ACCENT,
            font=(FONT, 10), wraplength=260, justify="left",
        ).pack(anchor="w", padx=14, pady=8)

    def _build_right(self, p):
        p.rowconfigure(0, weight=1)
        p.columnconfigure(0, weight=1)
        self._img_lbl = tk.Label(
            p, bg="#000000",
            text="Waiting for HoloLens stream…",
            fg="#444444", font=(FONT, 18),
        )
        self._img_lbl.grid(row=0, column=0, sticky="nsew")

    # helpers
    def _section(self, parent, title):
        f = tk.LabelFrame(
            parent, text=f"  {title}  ",
            bg=PANEL, fg=FG, font=(FONT, 10, "bold"),
            bd=1, relief="groove", labelanchor="nw",
        )
        f.pack(fill="x", padx=8, pady=6)
        return f

    def _btn(self, parent, label, cmd, color=BTN):
        """Label-based button — macOS respects bg/fg on Labels, not on tk.Button."""
        hover = _lighten(color)
        lbl = tk.Label(
            parent, text=label,
            bg=color, fg="white",
            font=(FONT, 11), anchor="w",
            padx=10, pady=6, cursor="hand2",
        )
        lbl.pack(fill="x", padx=10, pady=2)
        lbl.bind("<Enter>",    lambda e: lbl.config(bg=hover))
        lbl.bind("<Leave>",    lambda e: lbl.config(bg=color))
        lbl.bind("<Button-1>", lambda e: cmd())

    # ─────────────────────────────────────────────────────────────────── #
    # Helpers
    # ─────────────────────────────────────────────────────────────────── #

    def _prefix(self) -> str:
        """Return  name_YYYYMMDD_HHMMSS  (used as filename prefix)."""
        name = self._name_var.get().strip() or "participant"
        return f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _path(self, filename: str) -> str:
        return os.path.join(DATA_DIR, filename)

    def _send_cmd(self, cmd: str) -> bool:
        ip = self._nm.hololens_ip
        if not ip:
            self._conn_var.set("HoloLens not discovered yet")
            return False
        self._cmd_sock.sendto(cmd.encode(), (ip, CMD_PORT))
        print(f"[App] → {cmd}  ({ip}:{CMD_PORT})")
        return True

    # ─────────────────────────────────────────────────────────────────── #
    # Tracking commands
    # ─────────────────────────────────────────────────────────────────── #

    def _start_eye(self):
        if self._send_cmd("CMD:START_EYE"):
            self._eye_var.set("Eye tracking: recording ●")

    def _stop_eye(self):
        if self._send_cmd("CMD:STOP_EYE"):
            self._eye_var.set("Eye tracking: stopped ■")

    def _start_head(self):
        if self._send_cmd("CMD:START_HEAD"):
            self._head_var.set("Head tracking: recording ●")

    def _stop_head(self):
        if self._send_cmd("CMD:STOP_HEAD"):
            self._head_var.set("Head tracking: stopped ■")

    def _receive_files(self):
        prefix    = self._prefix()
        eye_path  = self._path(f"{prefix}_eye.csv")
        head_path = self._path(f"{prefix}_head.csv")

        # Start TCP listeners BEFORE sending the commands so HoloLens
        # can connect immediately when it receives CMD:SEND_*
        threading.Thread(
            target=self._nm.receive_file,
            args=(EYE_FILE_PORT, eye_path), daemon=True,
        ).start()
        threading.Thread(
            target=self._nm.receive_file,
            args=(HEAD_FILE_PORT, head_path), daemon=True,
        ).start()

        time.sleep(0.3)   # give servers time to bind

        self._send_cmd("CMD:SEND_EYE")
        self._send_cmd("CMD:SEND_HEAD")

        self._eye_var.set(f"Eye → {os.path.basename(eye_path)}")
        self._head_var.set(f"Head → {os.path.basename(head_path)}")

    # ─────────────────────────────────────────────────────────────────── #
    # Audio recording
    # ─────────────────────────────────────────────────────────────────── #

    def _start_audio(self):
        if not _AUDIO_OK:
            self._audio_var.set("Install: pip install sounddevice soundfile")
            return
        if self._recording:
            return

        self._recording    = True
        self._audio_chunks = []
        self._audio_start  = datetime.now()

        def _cb(indata, frames, t, status):
            self._audio_chunks.append(indata.copy())

        self._audio_stream = sd.InputStream(
            samplerate=self._sample_rate, channels=1, callback=_cb)
        self._audio_stream.start()
        self._audio_var.set(
            f"● Recording since {self._audio_start.strftime('%H:%M:%S')}")

    def _stop_audio(self):
        if not self._recording:
            return
        self._recording = False
        self._audio_stream.stop()
        self._audio_stream.close()

        name     = self._name_var.get().strip() or "participant"
        ts       = self._audio_start.strftime("%Y%m%d_%H%M%S")
        pref     = f"{name}_{ts}"
        end_time = datetime.now()

        wav_path = self._path(f"{pref}_audio.wav")
        log_path = self._path(f"{pref}_audio_log.csv")

        audio = np.concatenate(self._audio_chunks, axis=0)
        sf.write(wav_path, audio, self._sample_rate)

        with open(log_path, "w") as f:
            f.write("key,value\n")
            f.write(f"participant,{name}\n")
            f.write(f"start_utc,{self._audio_start.isoformat()}\n")
            f.write(f"end_utc,{end_time.isoformat()}\n")
            f.write(
                f"duration_s,"
                f"{(end_time - self._audio_start).total_seconds():.3f}\n")
            f.write(f"sample_rate,{self._sample_rate}\n")
            f.write(f"channels,1\n")
            f.write(f"file,{wav_path}\n")

        self._audio_var.set(f"Saved: {os.path.basename(wav_path)}")

    # ─────────────────────────────────────────────────────────────────── #
    # Frame receive → YOLO infer → label send → display
    # ─────────────────────────────────────────────────────────────────── #

    def _frame_loop(self):
        while True:
            frame = self._nm.receive_image()
            if frame is None:
                continue

            if self._model:
                try:
                    results    = self._model.predict(frame, conf=0.65, imgsz=640, verbose=False)
                    detections = _build_detections(results)
                    self._nm.send_label("udp", json.dumps(detections).encode())
                except Exception as e:
                    print(f"[App] Inference error: {e}")

            self._show_frame(frame)

    def _show_frame(self, frame):
        rw = self._img_lbl.winfo_width()
        rh = self._img_lbl.winfo_height()
        if rw < 10 or rh < 10:
            rw, rh = 950, 700

        h, w  = frame.shape[:2]
        scale = min(rw / w, rh / h)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img   = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._img_lbl.after(0, self._set_img, img)

    def _set_img(self, img):
        self._img_lbl.config(image=img, text="")
        self._img_lbl.image = img   # keep reference

    # ─────────────────────────────────────────────────────────────────── #
    # Status polling
    # ─────────────────────────────────────────────────────────────────── #

    def _poll_connection(self):
        ip = self._nm.hololens_ip
        self._conn_var.set(
            f"HoloLens: {ip}" if ip else "Searching for HoloLens…")
        self.root.after(2000, self._poll_connection)


# ──────────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    root = tk.Tk()
    ControllerApp(root)
    root.mainloop()
