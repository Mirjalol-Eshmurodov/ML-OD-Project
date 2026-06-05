# 🚗🧍 ML Object Detection Project

Real-time **car counting** and **people counting** using YOLOv8 and object tracking on video footage.

---

## 📌 Overview

This project implements two computer vision pipelines built on top of YOLOv8:

- **Car Counter** — detects and counts vehicles crossing a defined line in a video
- **People Counter** — detects pedestrians and tracks directional flow (Up / Down) using a virtual mask and crossing line

---

## 🎯 Results

| Module | Result |
|---|---|
| Car Counter | ~70 vehicles counted on test video |
| People Counter | 4 Up / 2 Down (directional tracking) |

---

## 🛠️ Tech Stack

- [YOLOv8](https://github.com/ultralytics/ultralytics) — object detection
- [OpenCV](https://opencv.org/) — video processing
- [cvzone](https://github.com/cvzone/cvzone) — visualization utilities
- [SORT](https://github.com/abewley/sort) (via `filterpy`, `lap`) — object tracking
- Python 3.10+

---

## 📁 Project Structure

```
ML-OD-Project/
├── car-counter/        # Vehicle counting module
├── people-counter/     # Pedestrian counting module with directional tracking
├── src/                # Shared utilities
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Mirjalol-Eshmurodov/ML-OD-Project.git
cd ML-OD-Project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download YOLOv8 model
```bash
# Automatically downloaded on first run, or manually:
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
```

### 4. Add your video
Place your input video in the `data/` folder (not included in repo).

### 5. Run
```bash
# Car counter
python car-counter/main.py

# People counter
python people-counter/main.py
```

---

## ⚙️ How It Works

**Car Counter:**
A virtual counting line is drawn across the video frame. Each detected vehicle that crosses the line is counted using SORT tracker to avoid double-counting.

**People Counter:**
A mask is applied to focus detection on a specific region. Pedestrians crossing the line are tracked by direction — counted separately as **Up** or **Down** based on their movement trajectory.

---

## 📦 Requirements

See [`requirements.txt`](requirements.txt). Key dependencies:
- `ultralytics==8.2.0`
- `opencv-python==4.10.0.84`
- `torch==2.3.1`
- `filterpy==1.4.5`
- `cvzone==1.5.6`
