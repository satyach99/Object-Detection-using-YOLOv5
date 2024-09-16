# Object Detection using YOLOv5

## Project Overview

This project implements object detection using the **YOLOv5** (You Only Look Once) deep learning model. YOLOv5 is a state-of-the-art real-time object detection system capable of detecting multiple objects in an image or video feed with high accuracy and speed. This project applies YOLOv5 to detect objects in a video and display the detection results with bounding boxes and labels.

## Features
- Real-time object detection using the YOLOv5 pre-trained model.
- Detects multiple objects in a video stream with bounding boxes and confidence scores.
- Utilizes PyTorch for model inference and OpenCV for video processing and visualization.

## Prerequisites

Ensure the following are installed before running the project:
- **Python 3.6+**
- **PyTorch** (for running the YOLOv5 model)
- **OpenCV** (for video capture and display)
- **YOLOv5** (from Ultralytics)

## Installation

1. **Clone the YOLOv5 repository**:
   ```bash
   git clone https://github.com/ultralytics/yolov5
   cd yolov5
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install additional libraries**:
   ```bash
   pip install opencv-python
   ```

## How to Run the Project

1. **Prepare the video file**:  
   Place your video file in the `data/` directory and name it `video.mp4`, or update the path in the Python script to point to the correct video file.

2. **Run the Python Script**:
   Execute the object detection script:
   ```bash
   python yolov5_object_detection.py
   ```

3. **Real-time Object Detection**:
   The script will process the video, apply YOLOv5 to detect objects in each frame, and display the results with bounding boxes and labels.

## Code Explanation

The key steps in the code include:
- **Model Loading**:  
  The YOLOv5 model is loaded using the PyTorch `torch.hub` functionality.
  ```python
  model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
  ```

- **Video Capture**:  
  The video is processed frame-by-frame using OpenCV.
  ```python
  cap = cv2.VideoCapture('data/video.mp4')
  ```

- **Object Detection**:  
  For each frame, the YOLOv5 model detects objects and returns bounding boxes, class labels, and confidence scores.
  ```python
  results = model(frame)
  ```

- **Bounding Box and Label Drawing**:  
  The bounding boxes and labels are drawn on the frame for each detected object.
  ```python
  cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
  cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
  ```

- **Display the Output**:  
  The processed frames are displayed in real-time with the detection results.
  ```python
  cv2.imshow('YOLOv5 Object Detection', frame)
  ```

## Results

The script will display the video with real-time object detection. Each object will be enclosed in a bounding box with a label indicating the object type and confidence score.

## Model Used

The project uses the **YOLOv5s** model, which is the small and fast version of the YOLOv5 family. You can experiment with other versions like YOLOv5m, YOLOv5l, or YOLOv5x for higher accuracy but at the cost of speed.

## Future Improvements

- Use **live camera feed** instead of video files for real-time object detection from webcams.
- Explore **custom model training** by fine-tuning YOLOv5 on a different dataset to detect specific objects.
- Integrate the system into an end-to-end application using **Flask** or **Streamlit** for real-time object detection in a web interface.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
