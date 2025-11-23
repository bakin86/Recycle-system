"""
Enhanced Trash Detection System - Camera-Based Recyclable/Non-Recyclable Classifier
OPTIMIZED FOR HIGH FPS with CLEAR CLASSIFICATION

Features:
- Real-time camera detection
- Clear GREEN (Recyclable) vs RED (Non-Recyclable) classification
- Human detection filter to reduce false positives
- Performance optimizations for smooth FPS
- Detailed statistics and visual feedback
"""

import cv2
import math
import cvzone
from ultralytics import YOLO
import time
import threading
from collections import deque
import numpy as np
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk


class RecyclableDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("♻️ Recyclable vs Non-Recyclable Detector - Camera System")
        self.root.geometry("1400x900")
        
        # Detection parameters
        self.model_path = "C:/Users/User/Desktop/GARBAGE/models/best.pt"
        self.confidence = 0.20
        self.enable_human_filter = True
        self.human_detection_confidence = 0.3
        self.device = 'cpu'
        
        # Performance settings
        self.process_every_n_frames = 1
        self.human_detect_every_n_frames = 5
        self.resize_for_detection = True
        self.detection_size = 640
        
        # State variables
        self.is_running = False
        self.is_paused = False
        self.cap = None
        self.current_frame = None
        self.detection_thread = None
        self.last_frame = None
        self.last_processed_frame = None
        
        # Statistics
        self.session_recyclable = 0
        self.session_non_recyclable = 0
        self.session_filtered = 0
        self.current_recyclable = 0
        self.current_non_recyclable = 0
        self.fps_history = deque(maxlen=30)
        
        # Models
        self.model = None
        self.human_model = None
        
        # Enhanced recyclable items dictionary with categories
        self.recyclable_categories = {
            # Metals
            'metals': [
                'aluminium foil', 'aluminium blister pack', 'metal bottle cap',
                'aerosol', 'drink can', 'food can', 'aluminium', 'aluminum',
                'metal', 'scrap metal', 'pop tab', 'tin can', 'steel can'
            ],
            # Paper products
            'paper': [
                'cardboard', 'paper', 'magazine paper', 'newspaper',
                'normal paper', 'paper bag', 'wrapping paper', 'office paper',
                'corrugated cardboard', 'paperboard'
            ],
            # Plastics (recyclable types)
            'plastic': [
                'clear plastic bottle', 'plastic bottle cap', 'plastic bottle',
                'plastic container', 'tupperware', 'hdpe', 'pet bottle',
                'plastic jug', 'detergent bottle'
            ],
            # Glass
            'glass': [
                'glass bottle', 'glass jar', 'glass cup', 'wine bottle',
                'beer bottle', 'glass container'
            ],
            # Cartons
            'cartons': [
                'drink carton', 'milk carton', 'juice carton', 'tetra pak'
            ]
        }
        
        # Non-recyclable items (common trash)
        self.non_recyclable_items = [
            'styrofoam', 'polystyrene', 'plastic bag', 'plastic film',
            'food waste', 'organic waste', 'cigarette', 'straw',
            'disposable plastic cup', 'plastic lid', 'chip bag',
            'candy wrapper', 'plastic cutlery', 'broken glass',
            'contaminated', 'dirty', 'greasy'
        ]
        
        self.setup_ui()
        self.load_models()
        
    def setup_ui(self):
        """Setup the main UI layout"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure custom colors
        style.configure('Recyclable.TLabel', foreground='green', font=('Arial', 11, 'bold'))
        style.configure('NonRecyclable.TLabel', foreground='red', font=('Arial', 11, 'bold'))
        
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        left_panel = ttk.Frame(main_container, width=320)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.setup_left_panel(left_panel)
        self.setup_right_panel(right_panel)
        
    def setup_left_panel(self, parent):
        """Setup control panel on the left"""
        
        # Title with recycling symbol
        title_frame = ttk.Frame(parent)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(title_frame, text="♻️ RECYCLING DETECTOR", 
                 font=("Arial", 16, "bold"), foreground="green").pack()
        ttk.Label(title_frame, text="Camera-Based Classification", 
                 font=("Arial", 10)).pack()
        ttk.Label(title_frame, text="🟢 GREEN = Recyclable", 
                 font=("Arial", 9), foreground="green").pack()
        ttk.Label(title_frame, text="🔴 RED = Non-Recyclable", 
                 font=("Arial", 9), foreground="red").pack()
        
        # Camera Control
        camera_frame = ttk.LabelFrame(parent, text="📷 Camera Control", padding=10)
        camera_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(camera_frame, text="Select Camera:").pack(anchor=tk.W)
        self.camera_var = tk.IntVar(value=0)
        camera_select = ttk.Frame(camera_frame)
        camera_select.pack(fill=tk.X, pady=5)
        ttk.Spinbox(camera_select, from_=0, to=5, textvariable=self.camera_var, 
                   width=8).pack(side=tk.LEFT)
        ttk.Label(camera_select, text="(0=default, 1=external)").pack(side=tk.LEFT, padx=5)
        
        ttk.Button(camera_frame, text="🎥 Open Camera", 
                  command=self.open_camera, width=25).pack(pady=5)
        
        # Detection Settings
        settings_frame = ttk.LabelFrame(parent, text="⚙️ Detection Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Confidence slider
        ttk.Label(settings_frame, text="Confidence Threshold:").pack(anchor=tk.W)
        conf_container = ttk.Frame(settings_frame)
        conf_container.pack(fill=tk.X, pady=(5, 10))
        
        self.confidence_var = tk.DoubleVar(value=self.confidence)
        self.confidence_slider = ttk.Scale(
            conf_container,
            from_=0.05,
            to=0.95,
            orient=tk.HORIZONTAL,
            variable=self.confidence_var,
            command=self.on_confidence_change
        )
        self.confidence_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.confidence_label = ttk.Label(conf_container, text="0.20", width=5)
        self.confidence_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Performance Mode
        ttk.Label(settings_frame, text="Performance Mode:").pack(anchor=tk.W, pady=(10, 0))
        self.perf_var = tk.StringVar(value="balanced")
        perf_frame = ttk.Frame(settings_frame)
        perf_frame.pack(fill=tk.X, pady=5)
        
        ttk.Radiobutton(perf_frame, text="⚡ Max Speed", variable=self.perf_var, 
                       value="fast", command=self.change_performance_mode).pack(anchor=tk.W)
        ttk.Radiobutton(perf_frame, text="⚖️ Balanced", variable=self.perf_var, 
                       value="balanced", command=self.change_performance_mode).pack(anchor=tk.W)
        ttk.Radiobutton(perf_frame, text="🎯 Max Quality", variable=self.perf_var, 
                       value="quality", command=self.change_performance_mode).pack(anchor=tk.W)
        
        # Human filter
        self.human_filter_var = tk.BooleanVar(value=self.enable_human_filter)
        ttk.Checkbutton(
            settings_frame,
            text="🚶 Filter Human Detections",
            variable=self.human_filter_var,
            command=self.toggle_human_filter
        ).pack(anchor=tk.W, pady=5)
        
        # Control Buttons
        control_frame = ttk.LabelFrame(parent, text="🎮 Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.start_btn = ttk.Button(control_frame, text="▶ Start Detection", 
                                    command=self.start_detection, width=25)
        self.start_btn.pack(pady=5)
        self.start_btn.config(state=tk.DISABLED)
        
        self.pause_btn = ttk.Button(control_frame, text="⏸ Pause", 
                                    command=self.toggle_pause, state=tk.DISABLED, width=25)
        self.pause_btn.pack(pady=5)
        
        self.stop_btn = ttk.Button(control_frame, text="⏹ Stop", 
                                   command=self.stop_detection, state=tk.DISABLED, width=25)
        self.stop_btn.pack(pady=5)
        
        ttk.Button(control_frame, text="🔄 Reset Statistics", 
                  command=self.reset_statistics, width=25).pack(pady=5)
        ttk.Button(control_frame, text="📸 Screenshot", 
                  command=self.save_screenshot, width=25).pack(pady=5)
        
        # Statistics Frame
        stats_frame = ttk.LabelFrame(parent, text="📊 Statistics", padding=10)
        stats_frame.pack(fill=tk.BOTH, expand=True)
        
        # FPS
        self.create_stat_row(stats_frame, "FPS:", "fps_label", "0", "black")
        
        # Current detections
        ttk.Label(stats_frame, text="Current Frame:", 
                 font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(10, 5))
        self.create_stat_row(stats_frame, "  🟢 Recyclable:", "current_recyclable_label", "0", "green")
        self.create_stat_row(stats_frame, "  🔴 Non-Recyclable:", "current_non_recyclable_label", "0", "red")
        
        # Session totals
        ttk.Separator(stats_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(stats_frame, text="Session Totals:", 
                 font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))
        self.create_stat_row(stats_frame, "  🟢 Recyclable:", "session_recyclable_label", "0", "green")
        self.create_stat_row(stats_frame, "  🔴 Non-Recyclable:", "session_non_recyclable_label", "0", "red")
        self.create_stat_row(stats_frame, "  🚶 Filtered:", "filtered_label", "0", "orange")
        
        # Status
        ttk.Separator(stats_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(stats_frame, text="Status:", font=("Arial", 9, "bold")).pack(anchor=tk.W)
        self.status_label = ttk.Label(stats_frame, text="Ready - Open camera to start", 
                                      foreground="gray", wraplength=280)
        self.status_label.pack(anchor=tk.W, pady=5)
        
    def create_stat_row(self, parent, label_text, var_name, initial_value, color):
        """Helper to create a statistics row"""
        container = ttk.Frame(parent)
        container.pack(fill=tk.X, pady=2)
        ttk.Label(container, text=label_text, width=18).pack(side=tk.LEFT)
        label = ttk.Label(container, text=initial_value, 
                         font=("Arial", 10, "bold"), foreground=color)
        label.pack(side=tk.LEFT)
        setattr(self, var_name, label)
        
    def setup_right_panel(self, parent):
        """Setup video display panel on the right"""
        video_frame = ttk.LabelFrame(parent, text="🎥 Live Camera Feed", padding=10)
        video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_canvas = tk.Canvas(video_frame, bg="black", width=1000, height=750)
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Instructions
        self.video_canvas.create_text(
            500, 300,
            text="♻️ RECYCLABLE DETECTOR",
            fill="white",
            font=("Arial", 24, "bold"),
            tags="placeholder"
        )
        self.video_canvas.create_text(
            500, 350,
            text="🎥 Click 'Open Camera' to start",
            fill="lightgray",
            font=("Arial", 14),
            tags="placeholder"
        )
        self.video_canvas.create_text(
            500, 400,
            text="🟢 GREEN = Recyclable Items",
            fill="lightgreen",
            font=("Arial", 12),
            tags="placeholder"
        )
        self.video_canvas.create_text(
            500, 430,
            text="🔴 RED = Non-Recyclable Items",
            fill="#FF6666",
            font=("Arial", 12),
            tags="placeholder"
        )
        
    def change_performance_mode(self):
        """Change performance mode"""
        mode = self.perf_var.get()
        
        if mode == "fast":
            self.process_every_n_frames = 2
            self.human_detect_every_n_frames = 6
            self.resize_for_detection = True
            self.detection_size = 480
            self.update_status("⚡ Max Speed mode - optimized for FPS")
            
        elif mode == "balanced":
            self.process_every_n_frames = 1
            self.human_detect_every_n_frames = 5
            self.resize_for_detection = True
            self.detection_size = 640
            self.update_status("⚖️ Balanced mode - good speed + accuracy")
            
        else:  # quality
            self.process_every_n_frames = 1
            self.human_detect_every_n_frames = 3
            self.resize_for_detection = False
            self.detection_size = 1280
            self.update_status("🎯 Max Quality mode - best accuracy")
        
    def load_models(self):
        """Load YOLO models"""
        try:
            self.update_status("Loading AI models...")
            
            if not os.path.exists(self.model_path):
                messagebox.showerror("Model Not Found", 
                    f"Trash detection model not found!\n\nExpected location:\n{self.model_path}\n\n"
                    "Please ensure the model file exists.")
                self.update_status("❌ Model not found!")
                return
            
            # Load trash detection model
            self.model = YOLO(self.model_path)
            
            # Load human detection model
            try:
                self.human_model = YOLO('yolov8n.pt')
            except:
                self.human_model = None
                messagebox.showwarning("Human Detection", 
                    "Could not load human detection model.\n"
                    "Human filtering will be disabled.")
            
            # Check for GPU
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = 'cuda'
                    self.model.to('cuda')
                    if self.human_model:
                        self.human_model.to('cuda')
                    device_text = "GPU (CUDA)"
                else:
                    device_text = "CPU"
            except:
                device_text = "CPU"
            
            self.update_status(f"✓ Models loaded! Using {device_text}")
            print(f"✓ Models loaded successfully")
            print(f"  Device: {device_text}")
            print(f"  Trash model: {self.model_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models:\n{str(e)}")
            self.update_status("❌ Failed to load models!")
    
    def open_camera(self):
        """Open camera for detection"""
        if self.is_running:
            messagebox.showwarning("Already Running", "Please stop current detection first")
            return
        
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None
            time.sleep(0.5)
        
        camera_index = self.camera_var.get()
        
        print(f"\n{'='*60}")
        print(f"🎥 Opening Camera {camera_index}...")
        print(f"{'='*60}")
        
        try:
            self.update_status(f"Opening camera {camera_index}...")
            
            # Try DirectShow (Windows)
            self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            time.sleep(1.0)
            
            if self.cap.isOpened():
                ret, test_frame = self.cap.read()
                if not ret or test_frame is None:
                    self.cap.release()
                    self.cap = None
            
            # Fallback to default
            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(camera_index)
                time.sleep(1.0)
                
                if not self.cap.isOpened():
                    raise Exception(
                        f"Could not open camera {camera_index}\n\n"
                        "Troubleshooting:\n"
                        "• Check camera connection\n"
                        "• Close other apps (Zoom, Skype, Teams)\n"
                        "• Try different camera index\n"
                        "• Check Windows camera permissions"
                    )
            
            # Optimize camera settings
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
            # Get actual settings
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            status_msg = f"✓ Camera {camera_index} ready!\nResolution: {width}x{height} @ {fps}fps"
            self.update_status(status_msg)
            self.start_btn.config(state=tk.NORMAL)
            
            print(f"✓ Camera {camera_index} opened successfully")
            print(f"  Resolution: {width}x{height}")
            print(f"  FPS: {fps}")
            print(f"{'='*60}\n")
            
        except Exception as e:
            if self.cap:
                try:
                    self.cap.release()
                except:
                    pass
                self.cap = None
            messagebox.showerror("Camera Error", str(e))
            self.update_status("❌ Failed to open camera")
            print(f"✗ Error: {e}\n")
    
    def start_detection(self):
        """Start detection process"""
        if self.model is None:
            messagebox.showerror("Error", "Models not loaded!")
            return
        
        if self.cap is None or not self.cap.isOpened():
            messagebox.showerror("Error", "No camera opened!")
            return
        
        self.is_running = True
        self.is_paused = False
        
        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)
        
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()
        
        self.update_status("🟢 Detection started!")
        print("\n🚀 Starting recyclable detection...")
    
    def detection_loop(self):
        """Main detection loop"""
        prev_time = time.time()
        frame_count = 0
        human_boxes = []
        
        while self.is_running:
            try:
                if self.is_paused:
                    time.sleep(0.1)
                    continue
                
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    print("Failed to read frame")
                    continue
                
                frame_count += 1
                original_frame = frame.copy()
                self.last_frame = original_frame
                
                # Calculate FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time
                self.fps_history.append(fps)
                
                # Process frames based on performance mode
                should_process = (frame_count % self.process_every_n_frames) == 0
                
                if should_process:
                    # Resize for detection if needed
                    if self.resize_for_detection:
                        h, w = frame.shape[:2]
                        scale = self.detection_size / max(h, w)
                        if scale < 1.0:
                            new_w = int(w * scale)
                            new_h = int(h * scale)
                            detection_frame = cv2.resize(frame, (new_w, new_h))
                            scale_back = (w / new_w, h / new_h)
                        else:
                            detection_frame = frame
                            scale_back = (1, 1)
                    else:
                        detection_frame = frame
                        scale_back = (1, 1)
                    
                    # Detect humans periodically
                    if frame_count % self.human_detect_every_n_frames == 0:
                        human_boxes = self.detect_humans(detection_frame, scale_back)
                    
                    # Run trash detection
                    results = self.model(
                        detection_frame,
                        stream=True,
                        verbose=False,
                        device=self.device,
                        half=True if self.device == 'cuda' else False
                    )
                    
                    # Process detections
                    original_frame, stats = self.process_detections(
                        original_frame, results, human_boxes, scale_back
                    )
                    
                    # Update statistics
                    self.current_recyclable = stats['recyclable']
                    self.current_non_recyclable = stats['non_recyclable']
                    self.session_recyclable += stats['recyclable']
                    self.session_non_recyclable += stats['non_recyclable']
                    self.session_filtered += stats['filtered']
                    
                    self.last_processed_frame = original_frame
                else:
                    if self.last_processed_frame is not None:
                        original_frame = self.last_processed_frame
                
                # Draw UI
                display_frame = self.draw_frame_ui(original_frame, fps, human_boxes)
                
                # Display and update
                self.display_frame(display_frame)
                self.update_statistics()
                
                time.sleep(0.001)
                
            except Exception as e:
                print(f"Error in detection loop: {e}")
                import traceback
                traceback.print_exc()
                break
        
        print("Detection loop ended")
        self.is_running = False
    
    def detect_humans(self, img, scale_back=(1, 1)):
        """Detect humans in frame"""
        if not self.enable_human_filter or self.human_model is None:
            return []
        
        try:
            results = self.human_model(img, stream=True, verbose=False, device=self.device)
            human_boxes = []
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if cls == 0 and conf >= self.human_detection_confidence:
                        x1, y1, x2, y2 = box.xyxy[0]
                        
                        # Scale back to original size
                        x1 = int(x1 * scale_back[0])
                        y1 = int(y1 * scale_back[1])
                        x2 = int(x2 * scale_back[0])
                        y2 = int(y2 * scale_back[1])
                        
                        margin = 20
                        human_boxes.append({
                            'x1': x1 - margin,
                            'y1': y1 - margin,
                            'x2': x2 + margin,
                            'y2': y2 + margin
                        })
            
            return human_boxes
        except:
            return []
    
    def box_overlap(self, box1, human_box):
        """Check if trash detection overlaps with human"""
        x1, y1, x2, y2 = box1
        hx1, hy1, hx2, hy2 = human_box['x1'], human_box['y1'], human_box['x2'], human_box['y2']
        
        x_overlap = max(0, min(x2, hx2) - max(x1, hx1))
        y_overlap = max(0, min(y2, hy2) - max(y1, hy1))
        overlap_area = x_overlap * y_overlap
        
        trash_area = (x2 - x1) * (y2 - y1)
        
        if trash_area > 0:
            overlap_ratio = overlap_area / trash_area
            return overlap_ratio > 0.3
        
        return False
    
    def is_recyclable(self, class_name):
        """Enhanced recyclable detection"""
        class_lower = class_name.lower()
        
        # Check non-recyclable first (priority)
        for non_recyclable in self.non_recyclable_items:
            if non_recyclable in class_lower:
                return False
        
        # Check recyclable categories
        for category, items in self.recyclable_categories.items():
            for recyclable in items:
                if recyclable in class_lower:
                    return True
        
        # Default keywords
        recyclable_keywords = ['bottle', 'can', 'jar', 'container']
        for keyword in recyclable_keywords:
            if keyword in class_lower and 'plastic bag' not in class_lower:
                return True
        
        return False
    
    def process_detections(self, img, results, human_boxes, scale_back=(1, 1)):
        """Process detections and draw on image"""
        recyclable_count = 0
        non_recyclable_count = 0
        filtered_count = 0
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                
                # Scale back to original size
                x1 = int(x1 * scale_back[0])
                y1 = int(y1 * scale_back[1])
                x2 = int(x2 * scale_back[0])
                y2 = int(y2 * scale_back[1])
                
                # Check human overlap
                is_filtered = False
                if self.enable_human_filter:
                    for human_box in human_boxes:
                        if self.box_overlap((x1, y1, x2, y2), human_box):
                            is_filtered = True
                            filtered_count += 1
                            break
                
                if is_filtered:
                    continue
                
                w, h = x2 - x1, y2 - y1
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                if conf >= self.confidence:
                    class_name = self.model.names[cls] if cls < len(self.model.names) else f"Class_{cls}"
                    recyclable = self.is_recyclable(class_name)
                    
                    if recyclable:
                        recyclable_count += 1
                        color = (0, 255, 0)  # GREEN
                        label = "♻️ RECYCLABLE"
                    else:
                        non_recyclable_count += 1
                        color = (0, 0, 255)  # RED
                        label = "🗑️ NON-RECYCLABLE"
                    
                    # Shorten class name if too long
                    display_name = class_name[:15] + "..." if len(class_name) > 15 else class_name
                    
                    # Draw bounding box with corners
                    cvzone.cornerRect(img, (x1, y1, w, h), l=15, t=3, colorR=color)
                    
                    # Draw label background
                    label_text = f'{label}: {display_name} ({conf:.2f})'
                    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # Position label above box
                    label_y = max(35, y1 - 10)
                    
                    # Draw filled rectangle for text background
                    cv2.rectangle(img, 
                                (x1, label_y - text_size[1] - 10), 
                                (x1 + text_size[0] + 10, label_y + 5),
                                color, -1)
                    
                    # Draw text
                    cv2.putText(img, label_text,
                              (x1 + 5, label_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        stats = {
            'recyclable': recyclable_count,
            'non_recyclable': non_recyclable_count,
            'filtered': filtered_count
        }
        
        return img, stats
    
    def draw_frame_ui(self, img, fps, human_boxes):
        """Draw UI elements on frame"""
        height, width = img.shape[:2]
        
        # Draw human boxes (yellow)
        for hbox in human_boxes:
            cv2.rectangle(img, (hbox['x1'], hbox['y1']), (hbox['x2'], hbox['y2']), 
                         (0, 255, 255), 2)
            cv2.putText(img, "HUMAN-FILTERED", (hbox['x1'], hbox['y1']-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw info panel
        overlay = img.copy()
        panel_height = 120
        cv2.rectangle(overlay, (10, 10), (400, panel_height), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
        
        # FPS with color coding
        fps_color = (0, 255, 0) if fps > 20 else (0, 255, 255) if fps > 10 else (0, 0, 255)
        cv2.putText(img, f"FPS: {int(fps)}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
        
        # Current detections
        cv2.putText(img, f"Recyclable: {self.current_recyclable}", (20, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, f"Non-Recyclable: {self.current_non_recyclable}", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Settings
        filter_status = "Filter: ON" if self.enable_human_filter else "Filter: OFF"
        cv2.putText(img, filter_status, (240, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f"Conf: {self.confidence:.2f}", (240, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img
    
    def display_frame(self, frame):
        """Display frame on canvas"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            frame_height, frame_width = frame_rgb.shape[:2]
            scale = min(canvas_width / frame_width, canvas_height / frame_height)
            
            new_width = int(frame_width * scale)
            new_height = int(frame_height * scale)
            
            frame_resized = cv2.resize(frame_rgb, (new_width, new_height), 
                                      interpolation=cv2.INTER_LINEAR)
            
            img_pil = Image.fromarray(frame_resized)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            
            self.video_canvas.delete("all")
            self.video_canvas.create_image(canvas_width // 2, canvas_height // 2, 
                                          image=img_tk, anchor=tk.CENTER)
            self.video_canvas.image = img_tk
    
    def update_statistics(self):
        """Update statistics labels"""
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        
        self.fps_label.config(text=f"{int(avg_fps)}")
        self.current_recyclable_label.config(text=str(self.current_recyclable))
        self.current_non_recyclable_label.config(text=str(self.current_non_recyclable))
        self.session_recyclable_label.config(text=str(self.session_recyclable))
        self.session_non_recyclable_label.config(text=str(self.session_non_recyclable))
        self.filtered_label.config(text=str(self.session_filtered))
    
    def reset_statistics(self):
        """Reset session statistics"""
        if self.is_running and not messagebox.askyesno("Reset Statistics", 
            "Reset statistics while detection is running?"):
            return
        
        self.session_recyclable = 0
        self.session_non_recyclable = 0
        self.session_filtered = 0
        self.update_statistics()
        self.update_status("Statistics reset")
    
    def toggle_pause(self):
        """Toggle pause state"""
        self.is_paused = not self.is_paused
        
        if self.is_paused:
            self.pause_btn.config(text="▶ Resume")
            self.update_status("⏸ Paused")
        else:
            self.pause_btn.config(text="⏸ Pause")
            self.update_status("🟢 Running...")
    
    def stop_detection(self):
        """Stop detection"""
        self.is_running = False
        self.is_paused = False
        
        if self.detection_thread:
            self.detection_thread.join(timeout=1.0)
        
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None
        
        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.DISABLED, text="⏸ Pause")
        self.stop_btn.config(state=tk.DISABLED)
        
        self.update_status("⏹ Detection stopped")
        print("\n⏹ Detection stopped\n")
    
    def save_screenshot(self):
        """Save current frame as screenshot"""
        if self.last_frame is None:
            messagebox.showwarning("No Frame", "No frame available to save")
            return
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"recyclable_detection_{timestamp}.jpg"
        
        cv2.imwrite(filename, self.last_frame)
        messagebox.showinfo("Screenshot Saved", f"Saved as:\n{filename}")
        self.update_status(f"Screenshot saved: {filename}")
    
    def on_confidence_change(self, value):
        """Handle confidence slider change"""
        self.confidence = float(value)
        self.confidence_label.config(text=f"{self.confidence:.2f}")
    
    def toggle_human_filter(self):
        """Toggle human filter"""
        self.enable_human_filter = self.human_filter_var.get()
        status = "enabled" if self.enable_human_filter else "disabled"
        self.update_status(f"🚶 Human filter {status}")
    
    def update_status(self, message):
        """Update status label"""
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_running:
            if messagebox.askokcancel("Quit", "Detection is running. Stop and quit?"):
                self.stop_detection()
                time.sleep(0.5)
                self.root.destroy()
        else:
            self.root.destroy()


def main():
    print("\n" + "="*60)
    print("♻️  RECYCLABLE vs NON-RECYCLABLE DETECTOR")
    print("   Camera-Based Real-Time Classification System")
    print("="*60)
    print("\nFeatures:")
    print("  🟢 GREEN boxes = Recyclable items")
    print("  🔴 RED boxes = Non-recyclable items")
    print("  🚶 Human detection filter (reduces false positives)")
    print("  ⚡ Performance optimization options")
    print("  📊 Real-time statistics tracking")
    print("\nStarting application...\n")
    
    root = tk.Tk()
    app = RecyclableDetectorGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()