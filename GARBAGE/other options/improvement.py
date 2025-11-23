"""
High-Performance Trash Detection with CVZone
Optimized for maximum FPS with adjustable confidence

Features:
- Multi-threading for better FPS
- Frame skipping option
- GPU acceleration
- Adjustable confidence in real-time
- Performance optimizations
"""

import cv2
import math
import cvzone
from ultralytics import YOLO
import time
import threading
from collections import deque


class HighFPSTrashDetector:
    def __init__(self, model_path, camera_index=0, confidence=0.20):
        """Initialize high-performance detector"""
        self.camera_index = camera_index
        self.confidence = confidence
        self.paused = False
        self.running = True
        
        # Performance settings
        self.skip_frames = 0  # Process every frame (0), or skip frames (1, 2, etc.)
        self.frame_count = 0
        self.use_gpu = True  # Set to False if no GPU
        
        # Initialize camera
        print("Initializing camera...")
        self.cap = cv2.VideoCapture(camera_index)
        
        # Optimize camera settings for FPS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 60)  # Request high FPS
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Fast codec
        
        if not self.cap.isOpened():
            raise Exception("Could not open camera!")
        
        print("✓ Camera opened!")
        
        # Load model
        print(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        
        # Set device
        if self.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    print("✓ Using GPU acceleration")
                    self.device = 'cuda'
                else:
                    print("✓ GPU not available, using CPU")
                    self.device = 'cpu'
            except:
                print("✓ Using CPU")
                self.device = 'cpu'
        else:
            self.device = 'cpu'
            print("✓ Using CPU")
        
        # Move model to device
        self.model.to(self.device)
        
        print("✓ Model loaded!")
        
        # Get class names from model
        self.classNames = self.model.names
        print(f"✓ Model has {len(self.classNames)} classes")
        
        # Statistics
        self.detection_count = 0
        self.total_detections = 0
        self.screenshot_count = 0
        
        # Frame buffer for threading
        self.frame_buffer = deque(maxlen=1)
        self.result_buffer = deque(maxlen=1)
        self.current_frame = None
        self.current_detections = []
        
        # FPS tracking
        self.fps_history = deque(maxlen=30)
        self.avg_fps = 0
        
    def capture_frames(self):
        """Thread function to capture frames"""
        while self.running:
            if not self.paused:
                ret, frame = self.cap.read()
                if ret:
                    self.frame_buffer.append(frame)
            time.sleep(0.001)  # Small delay to prevent CPU overuse
    
    def draw_ui(self, img, fps):
        """Draw optimized UI elements"""
        height, width = img.shape[:2]
        
        # Top bar - minimal for speed
        cv2.rectangle(img, (0, 0), (width, 80), (0, 0, 0), -1)
        
        # Title
        cv2.putText(img, "TRASH DETECTION - HIGH FPS", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # FPS with color coding
        fps_color = (0, 255, 0) if fps > 20 else (0, 255, 255) if fps > 10 else (0, 0, 255)
        cv2.putText(img, f"FPS: {int(fps)}", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_color, 2)
        
        # Confidence
        cv2.putText(img, f"Conf: {self.confidence:.2f}", (150, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Detection count
        cv2.putText(img, f"Objects: {self.detection_count}", (320, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 255), 2)
        
        # Average FPS
        cv2.putText(img, f"Avg: {int(self.avg_fps)}", (520, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Device indicator
        device_text = "GPU" if self.device == 'cuda' else "CPU"
        device_color = (0, 255, 0) if self.device == 'cuda' else (100, 100, 255)
        cv2.putText(img, device_text, (width - 100, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, device_color, 2)
        
        if self.paused:
            cv2.putText(img, "PAUSED", (width - 220, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Bottom controls - minimal
        cv2.rectangle(img, (0, height - 35), (width, height), (0, 0, 0), -1)
        controls = "Q:Quit | S:Save | +:Up -:Down | SPACE:Pause | 1/2/3:Presets | F:Skip"
        cv2.putText(img, controls, (10, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return img
    
    def process_detections(self, img, results):
        """Process and draw detections - optimized"""
        detection_count = 0
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                w, h = x2 - x1, y2 - y1
                
                # Get confidence and class
                conf = float(box.conf[0])
                conf_display = math.ceil(conf * 100) / 100
                cls = int(box.cls[0])
                
                # Only show if above confidence threshold
                if conf >= self.confidence:
                    detection_count += 1
                    
                    # Get class name
                    class_name = self.classNames[cls] if cls < len(self.classNames) else f"Class_{cls}"
                    
                    # Truncate long class names for speed
                    if len(class_name) > 15:
                        class_name = class_name[:12] + "..."
                    
                    # Draw detection - cvzone style
                    cvzone.cornerRect(img, (x1, y1, w, h), 
                                    l=15, t=2, colorR=(0, 255, 0))
                    
                    cvzone.putTextRect(img, f'{class_name} {conf_display}',
                                     (max(0, x1), max(35, y1)),
                                     scale=0.8, thickness=1,
                                     colorR=(0, 255, 0), colorT=(255, 255, 255))
        
        return img, detection_count
    
    def run(self):
        """Main high-performance detection loop"""
        print("\n" + "="*70)
        print("HIGH-PERFORMANCE MODE - OPTIMIZATIONS ACTIVE")
        print("="*70)
        print("CONTROLS:")
        print("  Q or ESC  - Quit")
        print("  S         - Save screenshot")
        print("  + or =    - Increase confidence (+0.05)")
        print("  - or _    - Decrease confidence (-0.05)")
        print("  SPACE     - Pause/Resume")
        print("  1         - Confidence 0.10 (max detection)")
        print("  2         - Confidence 0.20 (balanced)")
        print("  3         - Confidence 0.30 (conservative)")
        print("  F         - Toggle frame skip (boost FPS)")
        print("  G         - Toggle GPU/CPU")
        print("="*70)
        print("\nStarting detection...\n")
        
        # Start frame capture thread
        capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        capture_thread.start()
        
        prev_time = time.time()
        frame_skip_counter = 0
        
        try:
            while self.running:
                # Get frame from buffer
                if len(self.frame_buffer) > 0:
                    img = self.frame_buffer[-1].copy()
                    self.current_frame = img.copy()
                elif self.current_frame is not None:
                    img = self.current_frame.copy()
                else:
                    time.sleep(0.01)
                    continue
                
                # Calculate FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time
                self.fps_history.append(fps)
                self.avg_fps = sum(self.fps_history) / len(self.fps_history)
                
                # Frame skipping for FPS boost
                frame_skip_counter += 1
                should_process = (frame_skip_counter % (self.skip_frames + 1)) == 0
                
                if should_process and not self.paused:
                    # Run detection
                    results = self.model(
                        img, 
                        stream=True, 
                        verbose=False,
                        device=self.device,
                        half=True if self.device == 'cuda' else False  # FP16 for GPU
                    )
                    
                    # Process detections
                    img, detection_count = self.process_detections(img, results)
                    self.detection_count = detection_count
                    self.total_detections += detection_count
                
                # Draw UI
                img = self.draw_ui(img, fps)
                
                # Show frame
                cv2.imshow("Trash Detection - High FPS (CVZone)", img)
                
                # Handle keyboard (non-blocking)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:
                    print("\nQuitting...")
                    self.running = False
                    break
                
                elif key == ord('s'):
                    self.screenshot_count += 1
                    filename = f"detection_{self.screenshot_count}.jpg"
                    cv2.imwrite(filename, img)
                    print(f"✓ Screenshot: {filename}")
                
                elif key == ord('+') or key == ord('='):
                    self.confidence = min(0.95, self.confidence + 0.05)
                    print(f"Confidence: {self.confidence:.2f}")
                
                elif key == ord('-') or key == ord('_'):
                    self.confidence = max(0.05, self.confidence - 0.05)
                    print(f"Confidence: {self.confidence:.2f}")
                
                elif key == ord(' '):
                    self.paused = not self.paused
                    print("PAUSED" if self.paused else "RESUMED")
                
                elif key == ord('1'):
                    self.confidence = 0.10
                    print(f"Confidence: {self.confidence:.2f} (max detection)")
                
                elif key == ord('2'):
                    self.confidence = 0.20
                    print(f"Confidence: {self.confidence:.2f} (balanced)")
                
                elif key == ord('3'):
                    self.confidence = 0.30
                    print(f"Confidence: {self.confidence:.2f} (conservative)")
                
                elif key == ord('f'):
                    self.skip_frames = (self.skip_frames + 1) % 3
                    print(f"Frame skip: {self.skip_frames} (0=no skip, 1=skip 1, 2=skip 2)")
                
                elif key == ord('g'):
                    self.device = 'cpu' if self.device == 'cuda' else 'cuda'
                    self.model.to(self.device)
                    print(f"Device: {self.device}")
        
        except KeyboardInterrupt:
            print("\n\nStopped by user (Ctrl+C)")
            self.running = False
        
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            self.running = False
        
        finally:
            # Cleanup
            self.running = False
            time.sleep(0.1)  # Let thread finish
            self.cap.release()
            cv2.destroyAllWindows()
            
            # Print statistics
            print("\n" + "="*70)
            print("SESSION STATISTICS")
            print("="*70)
            print(f"Average FPS: {int(self.avg_fps)}")
            print(f"Total detections: {self.total_detections}")
            print(f"Screenshots: {self.screenshot_count}")
            print(f"Device used: {self.device}")
            print("="*70)
            print("\n✓ System closed. Goodbye!")


def main():
    """Main function"""
    
    # ========================================
    # CONFIGURATION
    # ========================================
    
    MODEL_PATH = "C:/Users/User/Desktop/GARBAGE/models/best.pt"
    CAMERA_INDEX = 0
    INITIAL_CONFIDENCE = 0.20
    
    # ========================================
    
    print("\n" + "="*70)
    print("INITIALIZING HIGH-FPS TRASH DETECTION")
    print("="*70)
    print(f"Model: {MODEL_PATH}")
    print(f"Camera: {CAMERA_INDEX}")
    print(f"Confidence: {INITIAL_CONFIDENCE}")
    print("="*70)
    print("\nOPTIMIZATIONS:")
    print("  ✓ Multi-threaded frame capture")
    print("  ✓ GPU acceleration (if available)")
    print("  ✓ FP16 inference (GPU only)")
    print("  ✓ Optimized camera settings")
    print("  ✓ Frame skipping option")
    print("  ✓ Minimal UI rendering")
    print("="*70)
    
    try:
        detector = HighFPSTrashDetector(MODEL_PATH, CAMERA_INDEX, INITIAL_CONFIDENCE)
        detector.run()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()