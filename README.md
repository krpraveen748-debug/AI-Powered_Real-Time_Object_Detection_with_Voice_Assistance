# Webcam Object Detection — YOLOv8

Real-time object detection from a live webcam feed using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and OpenCV. Detects a configurable subset of the 80 COCO classes (default: people, bags, suitcases, phones, teddy bears) and overlays bounding boxes plus a live object count on the video feed.

## Demo

Run the script, point your webcam at something, and watch it get boxed and labeled in real time — with a running object count in the top-left corner.

## Features

- Live detection using YOLOv8's nano model by default (fast, runs on CPU)
- Configurable confidence threshold
- Filter detection to specific object classes, or detect all 80 COCO classes
- On-screen object counter
- Clean quit with the `q` key

## Requirements

- Python 3.8+
- A webcam

## Installation

```bash
git clone https://github.com/<your-username>/yolo-webcam-detection.git
cd yolo-webcam-detection
pip install -r requirements.txt
```

## Usage

```bash
python detect.py
```

The YOLOv8 nano weights (`yolov8n.pt`) will be downloaded automatically on first run. Press **`q`** in the webcam window to quit.

## Configuration

All settings live at the top of `detect.py`:

| Variable | Description |
|---|---|
| `MODEL_SIZE` | YOLOv8 model variant: `n` (nano, fastest) through `x` (extra-large, most accurate) |
| `CONF_THRESHOLD` | Minimum confidence score (0–1) for a detection to be shown |
| `TARGET_CLASSES` | List of COCO class IDs to detect, or `None` for all 80 classes |

### Common COCO class IDs

| ID | Class |
|---|---|
| 0 | person |
| 24 | backpack |
| 26 | handbag |
| 28 | suitcase |
| 67 | cell phone |
| 77 | teddy bear |

The full list of 80 COCO class IDs is available in the [Ultralytics documentation](https://docs.ultralytics.com/datasets/detect/coco/).

## Tech Stack

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — object detection model
- [OpenCV](https://opencv.org/) — webcam capture and rendering

## License

MIT
