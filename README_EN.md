# Real-Time Hand Gesture Recognition

**Tech Stack:** Python, OpenCV, MediaPipe, NumPy, TensorFlow/TFLite, Docker

---

## Overview
A lightweight, real-time pipeline that estimates hand poses to recognize static signs and dynamic gestures. By converting hand landmarks into coordinate vectors, the system achieves high-speed inference suitable for edge devices.



## Key Features
* **Static & Temporal Recognition:** Classified via MLP for hand signs and LSTM for motion-based history.
* **Real-Time Performance:** Delivers **30+ FPS** using MediaPipe’s 21-keypoint extraction and TFLite.
* **On-the-Fly Data Logging:** Integrated system to record training data to CSV by pressing keys during live video.
* **Containerized Environment:** Ready-to-use Docker configuration for X11 visual forwarding.

---

## Installation & Usage

### 1. Requirements
* Python 3.7+
* MediaPipe 0.8.1+
* TensorFlow 2.3.0+

### 2. Run the Demo
```bash
# Direct run
python app.py

# Docker run (Linux/X11)
docker build -t hand_gesture .
xhost +local: && docker run --rm -it --device /dev/video0:/dev/video0 -e DISPLAY=$DISPLAY hand_gesture:latest
```

### **3. Training Custom Gestures**

1.  **Collect Data**: Run `app.py`.
    * Press **'k'** to log static hand signs (Keypoints).
    * Press **'h'** to log dynamic movement (Point History).
    * Press **0-9** while gesturing to save coordinates directly to the respective CSV files.
2.  **Train Model**: Open `keypoint_classification.ipynb` (for signs) or `point_history_classification.ipynb` (for motion) in Jupyter Notebook.
3.  **Export**: Execute all notebook cells to train the MLP/LSTM model and generate a new `.tflite` file. Ensure you update the label CSVs to match your new classes.



### **Directory Structure**

* `model/`: Contains pre-trained `.tflite` models and raw CSV datasets for keypoints and point history.
* `keypoint_classification.ipynb`: Dedicated notebook for training static hand sign classifiers.
* `point_history_classification.ipynb`: Dedicated notebook for training motion-based (LSTM) classifiers.
* `utils/cvfpscalc.py`: Helper module for real-time FPS performance tracking.
