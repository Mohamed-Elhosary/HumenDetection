import cv2
from ultralytics import YOLO

# Load your trained YOLOv8 model (or pre-trained model)
model = YOLO('yolov8s.pt')  # Replace with 'best.pt' if using your trained model

# Open the video file
video_path = 'test_video.mp4'  # Replace with the path to your video
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Optional: Save the output video
output_path = 'output_video_untrained.mp4'
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Perform inference on the frame
    results = model(frame)

    # Get detections for humans only (class 0 in COCO)
    human_detections = []
    for box in results[0].boxes:
        if box.cls[0] == 0:  # Class index 0 corresponds to 'person'
            human_detections.append(box)

    # Create a new image with only human detections drawn
    annotated_frame = frame.copy()
    for box in human_detections:
        # Extract box details
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        confidence = box.conf[0]               # Confidence score

        # Draw bounding box and label
        label = f'Person {confidence:.2f}'
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Human Detection', annotated_frame)

    # Write the frame to the output video
    out.write(annotated_frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
