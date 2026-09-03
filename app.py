"""
============================================================
  WEBCAM OBJECT DETECTION — YOLOv8 (Flask Web Version)
============================================================

Advanced real-time object detection via browser.
Access: http://localhost:5000

Features:
- Detects ALL 80 COCO classes
- Alternating red/black label colors
- Real-time statistics dashboard
- Object tracking & history
- Confidence filtering

Press Ctrl+C in terminal to stop the server.
============================================================
"""

from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import threading
from collections import defaultdict
from datetime import datetime
import time
import pyttsx3

app = Flask(__name__)

# ============================================================
# ⚙️  ADVANCED PRO CONFIG
# ============================================================
MODEL_SIZE = "n"        # n = fastest, x = most accurate (nano for speed)
CONF_THRESHOLD = 0.05   # Ultra-low threshold for maximum detection (5%)
IOU_THRESHOLD = 0.45    # Lower IOU = detect overlapping objects too

# Set to None to detect ALL 80 COCO classes
TARGET_CLASSES = None

# Color palette for alternating labels (Red/Black)
LABEL_COLORS = [
    (0, 0, 255),        # Red (BGR format)
    (0, 0, 0)           # Black
]

# Advanced settings
FRAME_SIZE = (1280, 720)  # Larger frame for better visibility
LABEL_FONT_SCALE = 1.2     # Larger font for labels
LABEL_THICKNESS = 3        # Thicker text
BOX_THICKNESS = 3          # Thicker bounding boxes
SKIP_FRAMES = 0            # Process every frame (0 = no skip for accuracy)
FRAME_DELAY = 100          # Delay in milliseconds between frames (100ms = 10 FPS for slower playback)
# ============================================================

# Global variables
cap = None
model = None
lock = threading.Lock()
detection_stats = {
    'total_detections': 0,
    'detection_history': [],
    'class_counts': defaultdict(int),
    'frames_processed': 0,
    'current_categories': [],
    'confidence_sum': 0.0,
    'last_alert': 'Monitoring scene',
    'start_time': datetime.now()
}

# ============================================================
# 🏷️  CLASS CATEGORY MAPPING (Generic grouping)
# ============================================================
# Maps COCO class IDs to generic categories
CLASS_CATEGORY_MAP = {
    # Person
    0: "person",
    
    # Animals (cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, teddy bear)
    14: "animals", 15: "animals", 16: "animals", 17: "animals", 18: "animals",
    19: "animals", 20: "animals", 21: "animals", 22: "animals", 75: "animals",
    
    # Food (banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake)
    45: "food", 46: "food", 47: "food", 48: "food", 49: "food",
    50: "food", 51: "food", 52: "food", 53: "food", 54: "food",
    
    # Electrical Items (TV, laptop, mouse, remote, keyboard, phone, microwave, oven, toaster, sink, refrigerator, clock, hair drier, toothbrush)
    62: "electrical items", 63: "electrical items", 64: "electrical items",
    65: "electrical items", 66: "electrical items", 67: "electrical items",
    68: "electrical items", 69: "electrical items", 70: "electrical items",
    71: "electrical items", 72: "electrical items", 74: "electrical items",
    78: "electrical items", 79: "electrical items",
    
    # Vehicles (bicycle, car, motorcycle, airplane, bus, train, truck, boat)
    1: "vehicles", 2: "vehicles", 3: "vehicles", 4: "vehicles", 
    5: "vehicles", 6: "vehicles", 7: "vehicles", 8: "vehicles",
    
    # Furniture (chair, couch, bed, dining table, bench, potted plant)
    13: "furniture", 55: "furniture", 56: "furniture", 57: "furniture",
    58: "furniture", 59: "furniture",
    
    # Sports & Accessories (backpack, umbrella, handbag, tie, suitcase, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket)
    23: "sports", 24: "sports", 25: "sports", 26: "sports", 27: "sports",
    29: "sports", 30: "sports", 31: "sports", 32: "sports", 33: "sports",
    34: "sports", 35: "sports", 36: "sports", 37: "sports",
    
    # Containers & Utensils (bottle, wine glass, cup, fork, knife, spoon, bowl, vase)
    38: "containers", 39: "containers", 40: "containers", 41: "containers",
    42: "containers", 43: "containers", 44: "containers", 73: "containers",
}

def get_category_name(class_id):
    """Convert COCO class ID to generic category name"""
    return CLASS_CATEGORY_MAP.get(class_id, "other")

# Text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Speed of speech
tts_engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
spoken_objects = {}  # Dictionary to track recently spoken objects (avoid repetition)
speech_lock = threading.Lock()

def speak_object_name(object_name):
    """Speak the detected object name in a separate thread - CONTINUOUS ANNOUNCEMENT"""
    def _speak():
        try:
            with speech_lock:
                tts_engine.say(object_name)
                tts_engine.runAndWait()
        except Exception as e:
            print(f"[SPEECH ERROR] {e}")
    
    # Allow continuous speech with minimal cooldown (0.25 seconds)
    # This will keep announcing items while they're being detected
    current_time = time.time()
    if object_name not in spoken_objects or (current_time - spoken_objects[object_name]) > 0.5:
        spoken_objects[object_name] = current_time
        # Speak in background thread
        speech_thread = threading.Thread(target=_speak, daemon=True)
        speech_thread.start()


def init_model():
    """Initialize YOLO model and webcam"""
    global cap, model
    
    print("[INFO] Loading model...")
    model = YOLO(f"yolov8{MODEL_SIZE}.pt")
    print("[INFO] Model loaded. Initializing webcam...")
    
    # DirectShow avoids the long initialization hang from the default backend on Windows.
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not access the webcam. Check it's connected and not in use by another app."
        )
    
    print("[INFO] Webcam initialized. Ready to stream.")


import time

def generate_frames():
    """Generate video frames with YOLO detections - ADVANCED PRO VERSION"""
    global cap, model, detection_stats
    
    if cap is None or model is None:
        return
    
    frame_count = 0
    
    while True:
        with lock:
            ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Resize for better visibility and faster processing
        frame = cv2.resize(frame, FRAME_SIZE)
        
        # Run YOLO detection with ultra-low confidence for ALL objects
        results = model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
        
        # Process detections with alternating colors
        detected_objects = []
        detection_stats['frames_processed'] += 1
        label_color_idx = 0
        
        for r in results:
            for box_idx, box in enumerate(r.boxes):
                class_id = int(box.cls[0])
                confidence = box.conf[0]
                
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Use the exact YOLO item name for display and speech
                class_name = r.names[class_id]
                
                # Update statistics with the exact item name
                detection_stats['total_detections'] += 1
                detection_stats['class_counts'][class_name] += 1
                detected_objects.append({
                    'name': class_name,
                    'confidence': float(confidence),
                    'timestamp': datetime.now().isoformat()
                })
                
                # 🔊 Pronounce the exact item name (for example, "cat" or "phone")
                speak_object_name(class_name)
                
                # Alternate colors for labels (Red and Black)
                label_color = LABEL_COLORS[label_color_idx % 2]
                label_color_idx += 1
                
                # Draw bounding box with thick GREEN lines
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), BOX_THICKNESS)
                
                # Create label text with the exact item name and confidence
                label = f"{class_name.upper()} {confidence:.0%}"
                
                # Get text size for better background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = LABEL_FONT_SCALE
                thickness = LABEL_THICKNESS
                text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                
                # Extended background for better visibility
                bg_x1 = max(0, x1 - 5)
                bg_y1 = max(0, y1 - text_size[1] - 15)
                bg_x2 = min(frame.shape[1], x1 + text_size[0] + 10)
                bg_y2 = min(frame.shape[0], y1 + 5)
                
                # Draw background rectangle (solid color)
                cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), label_color, -1)
                
                # Add white border for MAXIMUM contrast
                cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), 2)
                
                # Label text (white on colored background)
                cv2.putText(frame, label, (x1, y1 - 8),
                           font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        current_categories = sorted(set(obj['name'] for obj in detected_objects))
        detection_stats['current_categories'] = current_categories
        if 'person' in current_categories and 'animals' in current_categories:
            detection_stats['last_alert'] = 'Person and animals detected'
        elif len(current_categories) >= 3:
            detection_stats['last_alert'] = 'Multiple categories detected'
        elif current_categories:
            detection_stats['last_alert'] = f"Monitoring {', '.join(current_categories)}"
        else:
            detection_stats['last_alert'] = 'No categories detected'

        # Update history
        if detected_objects:
            detection_stats['detection_history'].append({
                'timestamp': datetime.now().isoformat(),
                'count': len(detected_objects),
                'objects': detected_objects
            })
            if len(detection_stats['detection_history']) > 100:
                detection_stats['detection_history'].pop(0)
        
        # Draw statistics panel on frame with LARGER, CLEARER text
        object_count = len(detected_objects)
        
        # Statistics background panel (LARGER)
        stat_bg_color = (30, 30, 30)  # Dark background
        cv2.rectangle(frame, (10, 10), (450, 150), stat_bg_color, -1)
        cv2.rectangle(frame, (10, 10), (450, 150), (0, 255, 0), 3)  # Green border
        
        # Statistics text - MUCH LARGER and CLEARER
        stats_y = 40
        line_height = 30
        
        # Title
        cv2.putText(frame, "DETECTION STATS", (20, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        # Objects in current frame
        cv2.putText(frame, f"Objects Now: {object_count}", (20, stats_y + line_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Total detections
        cv2.putText(frame, f"Total: {detection_stats['total_detections']}", (20, stats_y + line_height * 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        # Frame counter
        cv2.putText(frame, f"Frame: {detection_stats['frames_processed']}", (20, stats_y + line_height * 3),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        
        # Top-right corner - Mode info
        cv2.putText(frame, f"Threshold: {CONF_THRESHOLD*100:.0f}%", (FRAME_SIZE[0] - 320, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "ADVANCED PRO", (FRAME_SIZE[0] - 300, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        
        # Bottom banner - All detected classes
        if detected_objects:
            class_names = ", ".join(sorted(set([obj['name'] for obj in detected_objects])))
            cv2.rectangle(frame, (10, FRAME_SIZE[1] - 40), (FRAME_SIZE[0] - 10, FRAME_SIZE[1] - 10), 
                         (0, 100, 200), -1)
            cv2.putText(frame, f"Detected: {class_names}", (20, FRAME_SIZE[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Encode frame to JPEG with high quality
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame_bytes = buffer.tobytes()
        
        # Add frame delay for SLOWER playback (smooth 10 FPS)
        time.sleep(FRAME_DELAY / 1000.0)
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Stream video frames"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/stats')
def get_stats():
    """API endpoint for detection statistics"""
    with lock:
        # Calculate uptime
        uptime = (datetime.now() - detection_stats['start_time']).total_seconds()
        
        # Get top detected classes
        top_classes = sorted(
            detection_stats['class_counts'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return jsonify({
            'total_detections': detection_stats['total_detections'],
            'frames_processed': detection_stats['frames_processed'],
            'uptime_seconds': round(uptime, 2),
            'class_counts': dict(top_classes),
            'fps': round(detection_stats['frames_processed'] / max(uptime, 1), 1),
            'current_objects': len(detection_stats['current_categories']),
            'current_categories': detection_stats['current_categories'],
            'average_confidence': round(
                (sum(item['confidence'] for item in detection_stats['detection_history'][-1]['objects']) /
                 len(detection_stats['detection_history'][-1]['objects'])) * 100, 1
            ) if detection_stats['detection_history'] else 0,
            'alert': detection_stats['last_alert']
        })


@app.route('/api/reset', methods=['POST'])
def reset_stats():
    """Reset analytics without restarting the webcam or model."""
    with lock:
        detection_stats['total_detections'] = 0
        detection_stats['detection_history'].clear()
        detection_stats['class_counts'].clear()
        detection_stats['frames_processed'] = 0
        detection_stats['current_categories'] = []
        detection_stats['confidence_sum'] = 0.0
        detection_stats['last_alert'] = 'Monitoring scene'
        detection_stats['start_time'] = datetime.now()
    return jsonify({'status': 'reset'})


@app.route('/api/history')
def get_history():
    """API endpoint for detection history"""
    with lock:
        return jsonify({
            'history': detection_stats['detection_history'][-20:]  # Last 20 frames
        })


if __name__ == '__main__':
    try:
        init_model()
        print("\n" + "="*60)
        print("🚀 Server starting...")
        print("📺 Open your browser and go to: http://localhost:5000")
        print("="*60 + "\n")
        app.run(debug=False, host='localhost', port=5000, threaded=True)
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        if cap is not None:
            cap.release()
        print("\n[INFO] Webcam released. Server stopped.")
