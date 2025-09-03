import tkinter as tk
import socket
import networkmanager

class Utility():
    def get_local_ip_address(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                ip_address = s.getsockname()[0]  # Get the local IP address
            self.ip_address = ip_address
            self.ip_label.config(text=self.ip_address)
        except Exception as e:
            self.ip_label.config(text=f"Error: {e}")
    def set_unity_ip_address(self, ip):
        self.unity_ip_address = ip
        return
    def first_init_eye_tracking_receive(self):
        if self.is_first_call_eye:
            self.EyeTrackingNetworkManager = networkmanager.NetworkManager(protocol="tcp", 
                UNITY_IP="192.168.1.7", UNITY_PORT=self.eye_tracking_port, LISTEN_PORT=self.eye_tracking_port,
                no_split=True)
            self.is_first_call_eye = False
    def first_init_head_tracking_receive(self):
        if self.is_first_call_head:
            self.HeadTrackingNetworkManager = networkmanager.NetworkManager(protocol="tcp", 
                UNITY_IP="192.168.1.7", UNITY_PORT=self.head_tracking_port, LISTEN_PORT=self.head_tracking_port,
                no_split=True)
            self.is_first_call_head = False
    def receive_eye_tracking_data(self):
        self.first_init_eye_tracking_receive()
        self.EyeTrackingNetworkManager.receive_file(port=self.eye_tracking_port)
    def receive_head_tracking_data(self):
        self.first_init_head_tracking_receive()
        self.HeadTrackingNetworkManager.receive_file(port=self.head_tracking_port)
    def __init__(self):
        self.ip_address = ""
        self.unity_ip_address = ""
        self.eye_tracking_port = 5012
        self.head_tracking_port = 5013
        self.is_first_call_eye = True
        self.is_first_call_head = True
        window = tk.Tk()
        window.title("Utility")
        window.geometry("800x400")
        window.configure(bg="#FDFAF6")

        # creating widgets
        program_label = tk.Label(window, text="NutriMR",
            font=("Verdana", 24, "bold"),
            fg="#FF9B00",
            bg="#FDFAF6"
        )
        show_ip_label = tk.Label(window, text="Show IP address",
            font=("Courier", 16),
            fg="#333333",
            bg="#FDFAF6"
        )
        self.ip_label = tk.Label(window, text=self.ip_address, 
            font=("Courier", 16),
            fg="#333333",
            bg="#E4EFE7",
            width=12
        )
        get_local_ip_address_button = tk.Button(window, text="Get Local IP Address",
            command=lambda: self.get_local_ip_address(),
            highlightbackground="#FDFAF6"
        )
        enter_unity_ip_label = tk.Label(window, text="Enter Unity IP address",
            font=("Courier", 16),
            fg="#333333",
            bg="#FDFAF6"
        )
        unity_ip_entry = tk.Entry(window, width=12,
            font=("Courier", 16),
            bg="#E4EFE7",
        )
        set_unity_ip_button = tk.Button(window, text="Set Unity IP Address",
            command=lambda: self.set_unity_ip_address(unity_ip_entry.get()),
            highlightbackground="#FDFAF6"
        )
        receive_eye_tracking_data_label = tk.Label(window, text="Receive Eye Tracking Data",
            font=("Courier", 16),
            fg="#333333",
            bg="#FDFAF6"
        )
        receive_eye_tracking_data_button = tk.Button(window, text="Start Receiving Eye Tracking Data",
            command=self.receive_eye_tracking_data,
            highlightbackground="#FDFAF6"
        )
        receive_head_tracking_data_label = tk.Label(window, text="Receive Head Tracking Data",
            font=("Courier", 16),
            fg="#333333",
            bg="#FDFAF6"
        )
        receive_head_tracking_data_button = tk.Button(window, text="Start Receiving Head Tracking Data",
            command=self.receive_head_tracking_data,
            highlightbackground="#FDFAF6"
        )
        visualize_eye_tracking_data_label = tk.Label(window, text="Visualize Eye Tracking Data",
            font=("Courier", 16),
            fg="#333333",
            bg="#FDFAF6"
        )
        visualize_head_tracking_data_label = tk.Label(window, text="Visualize Head Tracking Data",
            font=("Courier", 16),
            fg="#333333",
            bg="#FDFAF6"
        )

        # placing widgets on the screen
        program_label.grid(row=0, column=0, columnspan=3, pady=20)
        show_ip_label.grid(row=1, column=0, sticky="w", padx=20, pady=10)
        self.ip_label.grid(row=1, column=1, sticky="nsew", padx=20, pady=10)
        get_local_ip_address_button.grid(row=1, column=2, sticky="e", padx=20, pady=10)
        enter_unity_ip_label.grid(row=2, column=0, sticky="w", padx=20, pady=10)
        unity_ip_entry.grid(row=2, column=1, sticky="nsew", padx=20, pady=10)
        set_unity_ip_button.grid(row=2, column=2, sticky="e", padx=20, pady=10)
        receive_eye_tracking_data_label.grid(row=3, column=0, sticky="w", padx=20, pady=10)
        receive_eye_tracking_data_button.grid(row=3, column=2, sticky="e", padx=20, pady=10)
        receive_head_tracking_data_label.grid(row=4, column=0, sticky="w", padx=20, pady=10)
        receive_head_tracking_data_button.grid(row=4, column=2, sticky="e", padx=20, pady=10)
        visualize_eye_tracking_data_label.grid(row=5, column=0, sticky="w", padx=20, pady=10)
        visualize_head_tracking_data_label.grid(row=6, column=0, sticky="w", padx=20, pady=10)

        window.attributes('-topmost', True)
        window.attributes('-topmost', False)
        window.focus_force()

        window.mainloop()

    
if __name__ == "__main__":
    utility = Utility()
