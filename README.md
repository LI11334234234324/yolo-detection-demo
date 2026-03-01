# Real-time Object Detection Demo

A beginner-level computer vision project using **YOLOv8** and **OpenCV**, built as a learning exercise related to AI-driven perception in robotics research.

## What It Does

- Opens your webcam (or a video file) and runs real-time object detection
- Draws bounding boxes with class labels and confidence scores
- Displays live FPS counter and object count
- Press `s` to save a screenshot, `q` to quit

## Demo

![demo](screenshot_example.jpg)

## Requirements

- Python 3.8+
- Webcam (or a `.mp4` video file)

## Installation

```bash
pip install opencv-python ultralytics
```

## Usage

```bash
python detect.py
```

To use a video file instead of webcam, edit the last line of `detect.py`:

```python
run_detection(source="your_video.mp4")
```

## How It Works

1. **YOLOv8** (You Only Look Once v8) is a state-of-the-art object detection model that processes the entire image in a single forward pass, making it fast enough for real-time use.
2. **OpenCV** captures frames from the camera and renders the annotated output.
3. The model (`yolov8n.pt`) is automatically downloaded on first run (~6MB, nano variant for speed).

## Relevance to Robotics Research

Object detection and perception are foundational components in robotics systems. In surgical robotics, accurate real-time perception of instruments and tissue is critical for autonomous operation — this project explores the basics of that pipeline.

## Next Steps (Ideas for Extension)

- [ ] Add object tracking across frames (ByteTrack)
- [ ] Experiment with custom dataset training for specific objects
- [ ] Integrate depth estimation for 3D perception
- [ ] Apply to endoscope/surgical video footage

## Author

李佳睿 — Undergraduate student, exploring robotics and computer vision.