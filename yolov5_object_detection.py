import torch
import cv2
import numpy as np

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Open the video file
cap = cv2.VideoCapture('data/video.mp4')

# Loop through the video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv5 model on the frame
    results = model(frame)

    # Extract bounding boxes, labels, and confidence scores
    boxes = results.xyxy[0].cpu().numpy()
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        label = f'{model.names[int(cls)]} {conf:.2f}'
        
        # Draw the bounding box and label on the frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Show the output frame
    cv2.imshow('YOLOv5 Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
