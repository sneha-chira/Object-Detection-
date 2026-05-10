from ultralytics import YOLO

def train_model():
    # Load a pre-trained model (recommended for faster training)
    # Options: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium), yolov8l.pt (large), yolov8x.pt (extra large)
    model = YOLO("yolov8n.pt")  
    
    print("Starting training...")
    
    # Train the model using the custom data.yaml
    # Adjust epochs, imgsz (image size), and batch_size according to your hardware capabilities
    results = model.train(
        data="data.yaml",
        epochs=50,       # Number of training epochs
        imgsz=640,       # Image size for training
        batch=16,        # Batch size (reduce if you run out of memory)
        name="custom_yolov8_model", # Name of the folder to save results
        device=""        # "" for auto (uses GPU if available, else CPU). Explicitly use "cpu" or "0" for GPU.
    )
    
    print("\nTraining complete!")
    print("Model weights are saved in: runs/detect/custom_yolov8_model/weights/best.pt")
    
    # Optional: Evaluate model performance on the validation set
    print("\nStarting validation...")
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")

if __name__ == "__main__":
    train_model()
