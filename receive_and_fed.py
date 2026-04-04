import json
import socket
import numpy as np
import cv2
from ultralytics import YOLO
import time
import argparse
import modelmanager
import tkinter as tk
from PIL import Image, ImageTk
from functools import partial

def parse_args():
    parser = argparse.ArgumentParser(description="Choose protocol to use (UDP or TCP).")
    parser.add_argument("--protocol", type=str, default="udp", choices=["udp", "tcp"], help="Protocol to use (udp or tcp)")
    parser.add_argument("--no_split", action="store_true", help="Disable splitting of image data into chunks")
    return parser.parse_args()

class App:
    def __init__(self, modelManager):
        self.modelManager = modelManager
        self.prev_time = time.time()
        self.latest_frame = None

        # Setup Tkinter
        self.root = tk.Tk()
        self.root.title("YOLO Real-Time Viewer")
        self.label = tk.Label(self.root)
        self.label.pack()

        # Example button (you can bind it to any function you want)
        self.btn = tk.Button(self.root, text="Refresh (TCP only)", 
                             command=partial(self.on_button_click, self.modelManager.networkManager))
        self.btn.pack(pady=10)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_frame(self):
        """Grab latest frame, run YOLO, and update Tkinter label in real time"""
        try:
            frame = self.modelManager.networkManager.receive_image()
            if frame is None:
                self.root.after(10, self.update_frame)
                return

            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - self.prev_time)
            self.prev_time = current_time

            # Run YOLO inference
            results = self.modelManager.do_inference(0.65, frame)
            detections = self.modelManager.build_detections(results)
            message = json.dumps(detections).encode("utf-8")
            print("Detections:", message)
            self.modelManager.networkManager.send_label("udp", message)

            # Annotate
            annotated = results[0].plot()
            cv2.putText(annotated, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Convert to Tkinter image
            img_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            self.label.config(image=imgtk)
            self.label.image = imgtk

        except socket.timeout:
            pass
        except Exception as e:
            print("Error:", e)

        # Keep looping
        self.root.after(10, self.update_frame)

    def on_button_click(self, networkManager):
        """Do something when button is pressed (example: save frame)"""
        # if self.latest_frame is not None:
        #     cv2.imwrite("saved_frame.jpg", self.latest_frame)
        #     print("Saved current frame!")
        networkManager.refresh()

    def on_close(self):
        self.modelManager.networkManager.self_destruct()
        self.root.destroy()

    def run(self):
        self.update_frame()
        self.root.mainloop()

def main(protocol, no_split):
    modelManager = modelmanager.ModelManager(protocol=protocol, no_split=no_split)
    app = App(modelManager)
    app.run()

if __name__ == "__main__":
    args = parse_args()
    protocol = args.protocol
    no_split = args.no_split
    main(protocol, no_split)
