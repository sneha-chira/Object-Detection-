import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO

# -----------------------------
# App Theme
# -----------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# -----------------------------
# Load YOLO Model
# -----------------------------
model = YOLO("yolov8n.pt")

# -----------------------------
# Main Window
# -----------------------------
root = ctk.CTk()
root.title("AI Object Detection System")
root.geometry("1400x800")
root.minsize(1200, 700)

# -----------------------------
# Variables
# ----------------------------
cap = None
running = False
loop_id = None

# -----------------------------
# Title
# -----------------------------
header = ctk.CTkLabel(
    root,
    text="Real-Time Object Detection Using YOLOv8",
    font=("Arial", 32, "bold")
)
header.pack(pady=20)

# -----------------------------
# Main Frame
# -----------------------------
main_frame = ctk.CTkFrame(root, corner_radius=20)
main_frame.pack(fill="both", expand=True, padx=20, pady=20)

# -----------------------------
# Left Control Panel
# -----------------------------
control_frame = ctk.CTkFrame(
    main_frame,
    width=300,
    corner_radius=20
)
control_frame.pack(side="left", fill="y", padx=20, pady=20)
control_frame.pack_propagate(False)
# -----------------------------
# Right Video Panel
# -----------------------------
video_frame = ctk.CTkFrame(
    main_frame,
    corner_radius=20
)
video_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

# -----------------------------
# Video Display
# -----------------------------
video_label = ctk.CTkLabel(video_frame, text="")
video_label.pack(expand=True)

# -----------------------------
# Status Label
# -----------------------------
status_label = ctk.CTkLabel(
    control_frame,
    text="Status: Ready",
    font=("Arial", 18)
)
status_label.pack(pady=20)

# -----------------------------
# Detection Counter
# -----------------------------
count_label = ctk.CTkLabel(
    control_frame,
    text="Objects Detected: 0",
    font=("Arial", 18)
)
count_label.pack(pady=10)

# -----------------------------
# Confidence Slider Label
# -----------------------------
conf_label = ctk.CTkLabel(
    control_frame,
    text="Confidence Threshold",
    font=("Arial", 16)
)
conf_label.pack(pady=(30, 10))

# -----------------------------
# Confidence Slider
# -----------------------------
confidence_value = ctk.DoubleVar(value=0.5)

confidence_slider = ctk.CTkSlider(
    control_frame,
    from_=0.1,
    to=1.0,
    variable=confidence_value
)
confidence_slider.pack(padx=20, fill="x")

# -----------------------------
# Functions
# -----------------------------
def start_camera():

    global cap, running

    stop_video()

    cap = cv2.VideoCapture(0)

    running = True

    status_label.configure(text="Status: Webcam Running")

    update_frame()
def upload_video():

    global cap, running

    stop_video()

    file_path = filedialog.askopenfilename(
        filetypes=[
            ("Video Files", "*.mp4 *.avi *.mov")
        ]
    )

    if file_path:

        cap = cv2.VideoCapture(file_path)

        running = True

        status_label.configure(text="Status: Video Running")

        update_frame()
def stop_video():

    global running, cap, loop_id

    running = False

    if loop_id is not None:
        video_label.after_cancel(loop_id)
        loop_id = None

    if cap:
        cap.release()

    status_label.configure(text="Status: Stopped")
def update_frame():

    global running, loop_id

    if running and cap:

        ret, frame = cap.read()

        if ret:

            confidence_threshold = confidence_value.get()

            results = model(frame)

            object_count = 0

            for result in results:

                boxes = result.boxes

                for box in boxes:

                    conf = float(box.conf[0])

                    if conf > confidence_threshold:
                        object_count += 1
            
            count_label.configure(
                text=f"Objects Detected: {object_count}"
            )

            annotated_frame = results[0].plot()

            rgb_frame = cv2.cvtColor(
                annotated_frame,
                cv2.COLOR_BGR2RGB
            )

            img = Image.fromarray(rgb_frame)

            img.thumbnail((900, 650))

            imgtk = ImageTk.PhotoImage(image=img)

            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)

            loop_id = video_label.after(10, update_frame)

        else:
            stop_video()

# -----------------------------
# Buttons
# -----------------------------
start_btn = ctk.CTkButton(
    control_frame,
    text="Start Webcam",
    command=start_camera,
    height=50,
    font=("Arial", 18, "bold"),
    corner_radius=15
)
start_btn.pack(padx=20, pady=20, fill="x")

upload_btn = ctk.CTkButton(
    control_frame,
    text="Upload Video",
    command=upload_video,
    height=50,
    font=("Arial", 18, "bold"),
    corner_radius=15
)
upload_btn.pack(padx=20, pady=20, fill="x")

stop_btn = ctk.CTkButton(
    control_frame,
    text="Stop Detection",
    command=stop_video,
    height=50,
    font=("Arial", 18, "bold"),
    corner_radius=15,
    fg_color="red",
    hover_color="#aa0000"
)
stop_btn.pack(padx=20, pady=20, fill="x")

exit_btn = ctk.CTkButton(
    control_frame,
    text="Exit Application",
    command=root.destroy,
    height=50,
    font=("Arial", 18, "bold"),
    corner_radius=15,
    fg_color="gray",
    hover_color="#444444"
)
exit_btn.pack(padx=20, pady=20, fill="x")

# -----------------------------
# Footer
# -----------------------------
footer = ctk.CTkLabel(
    root,
    text="Developed Using Python, OpenCV, YOLOv8 and CustomTkinter",
    font=("Arial", 14)
)
footer.pack(pady=10)

# -----------------------------
# Run App
# -----------------------------
root.mainloop()
# -----------------------------
# Cleanup
# -----------------------------
if cap:
    cap.release()

cv2.destroyAllWindows()