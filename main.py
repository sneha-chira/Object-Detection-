import cv2
from ultralytics import YOLO

# Load YOLO model (Using the custom trained model)
try:
    model = YOLO("runs/detect/custom_yolov8_model/weights/best.pt")
except Exception:
    model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Check webcam
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Press q to quit")

while True:

    # Read frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Detect objects
    results = model(frame)

    # Draw results on frame
    annotated_frame = results[0].plot()

    # Show output
    cv2.imshow("YOLO Object Detection", annotated_frame)

    # Exit on q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()