"""
Trash Detection with Camera
Based on your original code, modified for webcam use
"""

import cv2
import math
import cvzone
from ultralytics import YOLO

# Initialize camera (0 = default camera, 1 = external camera)
CAMERA_INDEX = 0
cap = cv2.VideoCapture(CAMERA_INDEX)

# Set camera resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera!")
    print("Try changing CAMERA_INDEX to 1 or 2")
    exit()

print("✓ Camera opened successfully!")
print("Press 'q' to quit")

# Load YOLO model with custom weights
MODEL_PATH = "C:/Users/User/Desktop/GARBAGE/models/best.pt"
model = YOLO(MODEL_PATH)

# Define class names (update these based on your model)
classNames = ['0', 'c', 'garbage', 'garbage_bag', 'sampah-detection', 'trash']

# FPS calculation
import time
prev_time = 0

while True:
    success, img = cap.read()
    
    if not success:
        print("Error: Failed to read from camera")
        break
    
    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    
    # Run detection
    results = model(img, stream=True)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Calculate width and height
            w, h = x2 - x1, y2 - y1
            
            # Get confidence and class
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            
            # Only show detections above confidence threshold
            if conf > 0.1:
                # Draw corner rectangle
                cvzone.cornerRect(img, (x1, y1, w, h), t=2)
                
                # Put text with class name and confidence
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', 
                                 (max(0, x1), max(35, y1)), 
                                 scale=1, thickness=1)
    
    # Display FPS on screen
    cv2.putText(img, f'FPS: {int(fps)}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the image
    cv2.imshow("Trash Detection - Camera", img)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
print("✓ Camera closed. Goodbye!")