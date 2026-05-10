import cv2
from ultralytics import YOLO

def detect_from_webcam():
    # Load your custom trained model
    # Replace the path below with your trained model's path if it's different
    model_path = "runs/detect/custom_yolov8_model/weights/best.pt"
    
    try:
        print(f"Attempting to load custom model from: {model_path}")
        model = YOLO(model_path)
        print("Successfully loaded custom model!")
    except Exception as e:
        print(f"Error loading custom model: {e}")
        print("Falling back to pretrained yolov8n.pt for demonstration purposes.")
        model = YOLO("yolov8n.pt")

    # Start webcam capture (0 is usually the default laptop webcam, change to 1 or 2 for external cameras)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\nStarting webcam feed. Press 'q' to quit.")

    while True:
        # Read a frame from the webcam
        success, frame = cap.read()
        
        if success:
            # Run YOLOv8 inference on the frame
            # conf=0.5 means it will only show detections with confidence > 50%
            results = model(frame, conf=0.5)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Custom Object Detection", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            print("Error: Failed to capture frame from webcam.")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_from_webcam()
