# 🎭 YOLOFaceCutter 🌟

YOLOFaceCutter is a powerful tool that enables easy video editing, focusing on specific faces.

## How to Use

1. **Dependencies Installation**:
   - Python 3.x
   - Install the required libraries by running:
     ```
     pip install onnxruntime gradio opencv-python-headless numpy torch Pillow face_recognition moviepy
     ```

2. **Setting Up**:
   - Clone the repository and navigate to the directory:
     ```
     git clone https://github.com/SpaceVX/YOLOFaceCutter.git
     cd YOLOFaceCutter
     ```
   - Run the program:
     ```
     python main.py
     ```

## How It Works

- YOLOFaceCutter utilizes the YOLO face detection model for identifying faces in videos.
- It preprocesses frames, applies the face detection model, and tracks faces throughout the video.
- Detected faces are saved and can be used for further editing or processing.

## Key Features

- 📹 Face Detection and Tracking: Automatically detects and tracks faces in videos.
- 🎨 Zoom and Save: Zooms in on detected faces and saves them as individual images for further processing.
- 📝 Logging: Logs information about detected faces and the processing pipeline.

These are the basic steps to utilize the program. Ensure you have installed all the necessary dependencies before running the program.
