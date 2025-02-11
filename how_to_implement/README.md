# Samurai Model Implementation for Object Tracking in Videos

## Overview
This project demonstrates how to use the Samurai model for object tracking in videos, leveraging YOLOv8 for object detection and SAM (Segment Anything Model) for segmentation. The pipeline includes:

1. Extracting frames from a video.
2. Detecting objects (e.g., a person or car) using YOLOv8.
3. Generating bounding box annotations for tracking.
4. Using the Samurai model to segment and track the object across frames.
5. Saving the output as a processed video.

## Installation and Setup

### 1. Check GPU Availability
The Samurai model requires a CUDA-compatible GPU. First, check for GPU support:

```bash
!nvidia-smi
```

Then, confirm that PyTorch is running on the correct device:

```python
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
```

### 2. Install Dependencies
Install the required Python libraries:

```bash
!pip install opencv-python supervision
!pip install tikzplotlib jpeg4py lmdb scipy loguru
```

### 3. Clone Samurai Implementation
Clone the custom implementation from GitHub:

```bash
!git clone https://github.com/ashrafulwork/samurai_implementation.git
```

Navigate into the repository and install additional dependencies:

```bash
%cd samurai_implementation
!pip install -e .
```

### 4. Download Samurai Checkpoints
Samurai requires pre-trained model weights. Set execution permissions and download them:

```bash
!chmod +x checkpoints/download_ckpts.sh
!cd checkpoints && ./download_ckpts.sh
```

## Processing the Video

### 5. Extract Frames from Video
Extract frames from an input video for processing:

```python
import cv2
import os

video_path = "/content/happy.mp4"  
output_folder = "/content/lmao/frames"
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  
    frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_filename, frame)
    frame_count += 1

cap.release()
print(f"Extracted {frame_count} frames and saved to {output_folder}")
```

### 6. Rename Frames for Samurai
Samurai requires frames to be numbered properly (e.g., `0001.jpg` instead of `frame_0001.jpg`). Rename them:

```python
import os

frames_folder = "/content/lmao/frames"

for filename in os.listdir(frames_folder):
    if filename.startswith("frame_") and filename.endswith(".jpg"):
        number = filename.split("_")[1].split(".")[0]
        new_filename = f"{int(number):04d}.jpg"
        os.rename(os.path.join(frames_folder, filename), os.path.join(frames_folder, new_filename))

print("✅ Frames renamed successfully!")
```

### 7. Verify Video Loading
Before proceeding, confirm that the video loads correctly:

```python
import cv2

video_path = "/content/happy.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video")
else:
    print("Video loaded successfully!")

cap.release()
```

## Object Detection with YOLOv8

### 8. Install YOLO and Decord
Install YOLOv8 (for object detection) and Decord (for efficient video reading):

```bash
!pip install decord ultralytics
```

### 9. Detecting the Object in the First Frame
Use YOLOv8 to detect objects in the first frame and extract the bounding box:

```python
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # 'yolov8s.pt' for better accuracy

video_path = "/content/happy.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Error loading video")
    exit()

results = model(frame)

PERSON_CLASS_ID = 0  # Class ID for "person" in COCO dataset

bbox = None
for detection in results[0].boxes.data:
    x1, y1, x2, y2, conf, cls = detection.tolist()
    if int(cls) == PERSON_CLASS_ID:
        bbox = (int(x1), int(y1), int(x2), int(y2))
        break  

if bbox is None:
    print("No person detected in the first frame.")
    exit()

# Save the bounding box
with open("bbox.txt", "w") as f:
    f.write(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}\n")

print(f"Bounding box saved: {bbox}")
```

## Using Samurai for Object Tracking

### 10. Running Samurai on Video Frames
Run Samurai with the detected bounding box file:

```bash
!python '/content/samurai_implementation/scripts/demo.py' \
  --video_path '/content/lmao/frames' \
  --txt_path '/content/samurai_implementation/bbox.txt' \
  --video_output_path '/content/zero2.mp4'
```

## Issues Faced & Solutions

### 1. Incorrect Bounding Box Format
**Issue:** Initially, the bounding box file (`bbox.txt`) was in an incorrect format, causing Samurai to fail.

**Solution:** The correct format is:

```
x_min,y_min,x_max,y_max
```

Fixed by correctly extracting the bounding box using YOLO.

### 2. Wrong Object Being Tracked
**Issue:** Samurai was tracking unintended objects because the bounding box file did not specify the object type.

**Solution:** Modified the YOLO detection script to explicitly track only the person class.

### 3. Input Video Was Causing Issues
**Issue:** Samurai was not working with certain videos due to encoding or frame rate issues.

**Solution:** Changing the input video resolved the issue.

### 4. Bounding Boxes Were Visible in Output
**Issue:** The output video displayed both segmentation masks and bounding boxes, while only segmentation was needed.

**Solution:** Modified the Samurai script to remove bounding box visualization.

---

## Acknowledgments
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Meta's Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
- [Samurai Implementation](https://github.com/ashrafulwork/samurai_implementation)

👨‍💻 Contributors

Special thanks to:

@yangchris11
@hananaq
