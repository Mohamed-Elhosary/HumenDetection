import os
from ultralytics import YOLO

def train_model():
    # Path to the YOLOv8 pre-trained model
    model_path = "yolov8s.pt"
    
    # Path to the data YAML file
    data_yaml_path = "D:/Year4/Fall/4907/Training/C2A_Dataset/data.yaml"
    
    # Initialize the YOLOv8 model
    model = YOLO(model_path)
    
    # Train the model
    model.train(
        data=data_yaml_path,  # Path to the data YAML file
        epochs=10,            # Number of training epochs
        imgsz=640,            # Image size for training
        batch=16,             # Batch size
        device=0,             # GPU device (use 'cpu' for CPU training or '0' for the first GPU)
        lr0=0.001,
        visualize=True,             
        project="Training",   # Folder where the training runs will be saved
        name="human_detection_2" # Subfolder name for this specific run
    )

    # Optional: Evaluate the model
    metrics = model.val()

    # Extract metrics
    mAP_50 = metrics.box.map50  # mAP@0.5
    mAP_50_95 = metrics.box.map  # mAP@[0.5:0.95]
    precision = metrics.box.precision
    recall = metrics.box.recall

    # Print metrics
    print(f"mAP@0.5: {mAP_50}")
    print(f"mAP@[0.5:0.95]: {mAP_50_95}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

if __name__ == '__main__':
    # Call the function to start the training process
    train_model()
