import face_recognition
import cv2
import numpy as np
import os
import datetime
import time
import math
from ultralytics import YOLO

# Load YOLOv8n (light) or switch to 'yolov8l.pt' for max accuracy if GPU available
model = YOLO('yolov8n.pt')

MATCH_THRESHOLD = 0.5
YOLO_CONFIDENCE_THRESHOLD = 0.5
FACE_DETECTION_MODEL = 'cnn'
video_capture = cv2.VideoCapture(0)

# Check camera
if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

# UI Colors and Settings
UI_COLORS = {
    'cyan': (255, 255, 0),
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'white': (255, 255, 255),
    'orange': (0, 165, 255),
    'yellow': (0, 255, 255),
    'purple': (255, 0, 255)
}

def draw_crosshair(frame, center_x, center_y, size=50):
    """Draw targeting crosshair"""
    color = UI_COLORS['cyan']
    thickness = 2
    
    # Horizontal lines
    cv2.line(frame, (center_x - size, center_y), (center_x - 20, center_y), color, thickness)
    cv2.line(frame, (center_x + 20, center_y), (center_x + size, center_y), color, thickness)
    
    # Vertical lines
    cv2.line(frame, (center_x, center_y - size), (center_x, center_y - 20), color, thickness)
    cv2.line(frame, (center_x, center_y + 20), (center_x, center_y + size), color, thickness)
    
    # Center dot
    cv2.circle(frame, (center_x, center_y), 3, color, -1)

def draw_corner_brackets(frame, x1, y1, x2, y2, color, thickness=2, length=20):
    """Draw corner brackets around detection"""
    # Top-left
    cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness)
    
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness)
    
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness)
    
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness)

def draw_scan_lines(frame, y_start, y_end, alpha=0.3):
    """Draw animated scan lines effect"""
    scan_time = time.time()
    scan_offset = int((scan_time * 100) % (y_end - y_start))
    
    overlay = frame.copy()
    for i in range(0, frame.shape[0], 4):
        if i % 8 == scan_offset % 8:
            cv2.line(overlay, (0, i), (frame.shape[1], i), UI_COLORS['cyan'], 1)
    
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_hud_overlay(frame, fps, target_count, known_faces):
    """Draw heads-up display overlay"""
    height, width = frame.shape[:2]
    overlay = frame.copy()
    
    # Semi-transparent overlay panels
    panel_color = (0, 0, 0)
    alpha = 0.7
    
    # Top status bar
    cv2.rectangle(overlay, (0, 0), (width, 60), panel_color, -1)
    
    # Side panels
    cv2.rectangle(overlay, (0, 60), (200, height), panel_color, -1)
    cv2.rectangle(overlay, (width-200, 60), (width, height), panel_color, -1)
    
    # Bottom status bar
    cv2.rectangle(overlay, (0, height-80), (width, height), panel_color, -1)
    
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
    
    # Status text
    cv2.putText(frame, "E.D.I.T.H SYSTEM ACTIVE", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, UI_COLORS['cyan'], 2)
    cv2.putText(frame, f"TARGETS: {target_count}", (10, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, UI_COLORS['green'], 2)
    
    # System info
    cv2.putText(frame, f"FPS: {fps:.1f}", (width-180, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, UI_COLORS['yellow'], 2)
    cv2.putText(frame, f"KNOWN: {len(known_faces)}", (width-180, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, UI_COLORS['green'], 2)
    
    # Timestamp
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, timestamp, (width-120, height-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, UI_COLORS['white'], 2)
    
    # System status
    cv2.putText(frame, "SURVEILLANCE MODE", (10, height-50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, UI_COLORS['cyan'], 2)
    cv2.putText(frame, "THREAT LEVEL: LOW", (10, height-25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, UI_COLORS['green'], 2)

def draw_target_info_panel(frame, name, distance, confidence, x, y):
    """Draw detailed target information panel"""
    panel_width = 200
    panel_height = 120
    
    # Adjust panel position if too close to edges
    if x + panel_width > frame.shape[1]:
        x = frame.shape[1] - panel_width - 10
    if y - panel_height < 0:
        y = panel_height + 10
    
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y-panel_height), (x+panel_width, y), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    # Border
    color = UI_COLORS['green'] if name != "Unknown" else UI_COLORS['red']
    cv2.rectangle(frame, (x, y-panel_height), (x+panel_width, y), color, 2)
    
    # Target info
    cv2.putText(frame, "TARGET DATA", (x+10, y-95), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, UI_COLORS['cyan'], 1)
    cv2.putText(frame, f"ID: {name}", (x+10, y-75), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, UI_COLORS['white'], 1)
    cv2.putText(frame, f"DIST: {distance}m", (x+10, y-55), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, UI_COLORS['yellow'], 1)
    cv2.putText(frame, f"CONF: {confidence:.1f}%", (x+10, y-35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, UI_COLORS['green'], 1)
    cv2.putText(frame, f"STATUS: {'IDENTIFIED' if name != 'Unknown' else 'UNKNOWN'}", 
                (x+10, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

def draw_radar_sweep(frame, center_x, center_y, radius=100):
    """Draw animated radar sweep"""
    sweep_time = time.time()
    angle = (sweep_time * 90) % 360  # Complete rotation every 4 seconds
    
    # Draw radar circles
    for r in range(radius//3, radius, radius//3):
        cv2.circle(frame, (center_x, center_y), r, UI_COLORS['cyan'], 1)
    
    # Draw sweep line
    end_x = int(center_x + radius * math.cos(math.radians(angle)))
    end_y = int(center_y + radius * math.sin(math.radians(angle)))
    cv2.line(frame, (center_x, center_y), (end_x, end_y), UI_COLORS['green'], 2)
    
    # Draw center
    cv2.circle(frame, (center_x, center_y), 3, UI_COLORS['red'], -1)

# Load known faces
def load_known_faces(folder):
    encodings, names = [], []
    for file in os.listdir(folder):
        if file.lower().endswith(('.jpg', '.png')):
            path = os.path.join(folder, file)
            img = face_recognition.load_image_file(path)
            enc = face_recognition.face_encodings(img, num_jitters=10)
            if enc:
                encodings.append(enc[0])
                names.append(os.path.splitext(file)[0])
    return encodings, names

known_face_encodings, known_face_names = load_known_faces("known_faces")
os.makedirs("unknown_faces", exist_ok=True)

# Distance estimation helper
FOCAL_LENGTH = 650
KNOWN_WIDTH_FACE = 0.16  # meters

def estimate_distance(perceived_width_px, known_width=KNOWN_WIDTH_FACE, focal_length=FOCAL_LENGTH):
    if perceived_width_px <= 0:
        return None
    distance = (known_width * focal_length) / perceived_width_px
    return round(max(0.3, min(distance, 10.0)), 2)

print("🚀 E.D.I.T.H Enhanced Tactical Interface Online. Press 'q' to terminate.")

prev_time = time.time()
frame_count = 0
target_acquired_sound = True  # Flag for sound alerts

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Frame acquisition failed.")
        break

    frame_count += 1
    original_frame = frame.copy()
    
    # Apply subtle bilateral filter
    frame = cv2.bilateralFilter(frame, 9, 75, 75)
    
    target_count = 0
    
    # YOLO Object Detection
    yolo_results = model(frame, verbose=False)[0]
    for box in yolo_results.boxes:
        conf = float(box.conf[0])
        if conf >= YOLO_CONFIDENCE_THRESHOLD:
            cls = int(box.cls[0])
            label = model.names[cls]
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            width_px = x2 - x1
            distance = estimate_distance(width_px, known_width=0.5)
            
            # Enhanced object detection display
            color = UI_COLORS['orange']
            draw_corner_brackets(frame, x1, y1, x2, y2, color, 2, 15)
            
            # Object info display
            cv2.putText(frame, f"{label.upper()}", (x1, y1 - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"CONF: {conf:.2f} | DIST: {distance}m", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            target_count += 1

    # Face Recognition
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame, model=FACE_DETECTION_MODEL)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []
    face_confidences = []

    for face_encoding in face_encodings:
        distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        name = "UNKNOWN"
        confidence = 0
        
        if len(distances) > 0:
            best_match = np.argmin(distances)
            if distances[best_match] < MATCH_THRESHOLD:
                name = known_face_names[best_match].upper()
                confidence = (1 - distances[best_match]) * 100
        
        face_names.append(name)
        face_confidences.append(confidence)

    # Enhanced Face Display
    for (top, right, bottom, left), name, confidence in zip(face_locations, face_names, face_confidences):
        # Scale back up face locations
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        width_px = right - left
        distance = estimate_distance(width_px)
        
        # Color coding
        color = UI_COLORS['green'] if name != "UNKNOWN" else UI_COLORS['red']
        
        # Enhanced face detection box
        draw_corner_brackets(frame, left, top, right, bottom, color, 3, 25)
        
        # Targeting crosshair on face center
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        draw_crosshair(frame, center_x, center_y, 30)
        
        # Target acquired indicator
        if name != "UNKNOWN":
            cv2.putText(frame, "TARGET ACQUIRED", (left, top - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, UI_COLORS['green'], 2)
        else:
            cv2.putText(frame, "ANALYZING...", (left, top - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, UI_COLORS['red'], 2)
        
        # Detailed info panel
        draw_target_info_panel(frame, name, distance, confidence, right + 10, bottom)
        
        # Save unknown faces
        if name == "UNKNOWN":
            folder = datetime.datetime.now().strftime("%Y-%m-%d")
            save_dir = os.path.join("unknown_faces", folder)
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%H%M%S_%f")
            crop = original_frame[top:bottom, left:right]
            save_path = os.path.join(save_dir, f"unknown_{timestamp}.jpg")
            cv2.imwrite(save_path, crop)
            print(f"💾 Unknown target archived: {save_path}")
        
        target_count += 1

    # FPS Calculation
    new_time = time.time()
    fps = 1 / (new_time - prev_time) if prev_time else 0
    prev_time = new_time
    
    # Draw scan lines effect
    draw_scan_lines(frame, 0, frame.shape[0], 0.1)
    
    # Draw HUD overlay
    draw_hud_overlay(frame, fps, target_count, known_face_names)
    
    # Draw mini radar in corner
    draw_radar_sweep(frame, frame.shape[1] - 100, 150, 60)
    
    # Main crosshair in center
    center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
    draw_crosshair(frame, center_x, center_y, 80)
    
    # Display enhanced frame
    cv2.imshow('🛡️ E.D.I.T.H Tactical Recognition System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 E.D.I.T.H System Shutting Down...")
        break

video_capture.release()
cv2.destroyAllWindows()
print("✅ System terminated successfully.")
