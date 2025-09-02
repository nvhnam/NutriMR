import tkinter as tk

def on_click(event=None):
    print("Button clicked!")

root = tk.Tk()
root.title("Red Button Window")
root.geometry("200x100")

canvas = tk.Canvas(root, width=100, height=40, highlightthickness=0)
canvas.pack(pady=20)

rect = canvas.create_rectangle(0, 0, 100, 40, fill="red", outline="red")
text = canvas.create_text(50, 20, text="Click Me", fill="white")

canvas.bind("<Button-1>", on_click)

root.mainloop()
