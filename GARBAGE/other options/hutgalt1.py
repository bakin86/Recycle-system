"""
Trash Detection System with Tkinter GUI - FIXED CAMERA VERSION
Color-coded: GREEN = Recyclable, RED = Non-Recyclable
WITH HUMAN DETECTION FILTER
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


class TrashDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Trash Detection System - AI Recycling Classifier")
        self.root.geometry("1400x900")
        
        # Detection parameters
        self.model_path = "C:/Users/User/Desktop/GARBAGE/models/best.pt"
        self.confidence = 0.20
        self.enable_human_filter = True
        self.human_detection_confidence = 0.3
        self.device = 'cpu'
        
        # State variables
        self.is_running = False
        self.is_paused = False
        self.current_mode = None
        self.cap = None
        self.current_frame = None
        self.detection_thread = None
        self.last_frame = None
        
        # Statistics
        self.total_detections = 0
        self.recyclable_count = 0
        self.non_recyclable_count = 0
        self.filtered_count = 0
        self.fps_history = deque(maxlen=30)
        
        # Models
        self.model = None
        self.human_model = None
        
        # Recyclable items list
        self.recyclable_items = {
            'Aluminium foil', 'Aluminium blister pack', 'Metal bottle cap', 
            'Aerosol', 'Drink can', 'Food Can', 'Cardboard', 'Paper', 
            'Magazine paper', 'Tissues', 'Wrapping paper', 'Normal paper', 
            'Paper bag', 'Clear plastic bottle', 'Plastic bottle cap', 
            'Plastic bottle', 'Drink carton', 'Disposable plastic cup', 
            'Plastic cup', 'Plastic lid', 'Tupperware', 'Plastic container',
            'Glass bottle', 'Broken glass', 'Glass jar', 'Glass cup',
            'Aluminium', 'Metal', 'Scrap metal', 'Pop tab', 'Battery',
        }
        
        self.setup_ui()
        self.load_models()
        
    def setup_ui(self):
        """Setup the main UI layout"""
        style = ttk.Style()
        style.theme_use('clam')
        
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        left_panel = ttk.Frame(main_container, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.setup_left_panel(left_panel)
        self.setup_right_panel(right_panel)
        
    def setup_left_panel(self, parent):
        """Setup control panel on the left"""
        
        # Title
        title_frame = ttk.Frame(parent)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(title_frame, text="🗑️ Trash Detection", font=("Arial", 18, "bold")).pack()
        ttk.Label(title_frame, text="AI-Powered Recycling Classifier", font=("Arial", 10)).pack()
        
        # Input Mode Selection
        mode_frame = ttk.LabelFrame(parent, text="Input Mode", padding=10)
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(mode_frame, text="📷 Open Camera", command=self.open_camera, width=25).pack(pady=5)
        ttk.Button(mode_frame, text="🎥 Load Video File", command=self.load_video, width=25).pack(pady=5)
        ttk.Button(mode_frame, text="🖼️ Load Image File", command=self.load_image, width=25).pack(pady=5)
        
        # Settings Frame
        settings_frame = ttk.LabelFrame(parent, text="Settings", padding=10)
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
        
        # Human filter toggle
        self.human_filter_var = tk.BooleanVar(value=self.enable_human_filter)
        ttk.Checkbutton(
            settings_frame,
            text="Enable Human Detection Filter",
            variable=self.human_filter_var,
            command=self.toggle_human_filter
        ).pack(anchor=tk.W, pady=5)
        
        # Camera selection
        ttk.Label(settings_frame, text="Camera Index:").pack(anchor=tk.W)
        self.camera_var = tk.IntVar(value=0)
        ttk.Spinbox(settings_frame, from_=0, to=5, textvariable=self.camera_var, width=10).pack(anchor=tk.W, pady=5)
        
        # Control Buttons
        control_frame = ttk.LabelFrame(parent, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.start_btn = ttk.Button(control_frame, text="▶ Start Detection", command=self.start_detection, width=25)
        self.start_btn.pack(pady=5)
        
        self.pause_btn = ttk.Button(control_frame, text="⏸ Pause", command=self.toggle_pause, state=tk.DISABLED, width=25)
        self.pause_btn.pack(pady=5)
        
        self.stop_btn = ttk.Button(control_frame, text="⏹ Stop", command=self.stop_detection, state=tk.DISABLED, width=25)
        self.stop_btn.pack(pady=5)
        
        ttk.Button(control_frame, text="📸 Save Screenshot", command=self.save_screenshot, width=25).pack(pady=5)
        
        # Statistics Frame
        stats_frame = ttk.LabelFrame(parent, text="Statistics", padding=10)
        stats_frame.pack(fill=tk.BOTH, expand=True)
        
        # FPS
        fps_container = ttk.Frame(stats_frame)
        fps_container.pack(fill=tk.X, pady=5)
        ttk.Label(fps_container, text="FPS:", width=15).pack(side=tk.LEFT)
        self.fps_label = ttk.Label(fps_container, text="0", font=("Arial", 10, "bold"))
        self.fps_label.pack(side=tk.LEFT)
        
        # Recyclable count
        recyclable_container = ttk.Frame(stats_frame)
        recyclable_container.pack(fill=tk.X, pady=5)
        ttk.Label(recyclable_container, text="🟢 Recyclable:", width=15, foreground="green").pack(side=tk.LEFT)
        self.recyclable_label = ttk.Label(recyclable_container, text="0", font=("Arial", 10, "bold"), foreground="green")
        self.recyclable_label.pack(side=tk.LEFT)
        
        # Non-recyclable count
        non_recyclable_container = ttk.Frame(stats_frame)
        non_recyclable_container.pack(fill=tk.X, pady=5)
        ttk.Label(non_recyclable_container, text="🔴 Non-Recyclable:", width=15, foreground="red").pack(side=tk.LEFT)
        self.non_recyclable_label = ttk.Label(non_recyclable_container, text="0", font=("Arial", 10, "bold"), foreground="red")
        self.non_recyclable_label.pack(side=tk.LEFT)
        
        # Total detections
        total_container = ttk.Frame(stats_frame)
        total_container.pack(fill=tk.X, pady=5)
        ttk.Label(total_container, text="Total Detected:", width=15).pack(side=tk.LEFT)
        self.total_label = ttk.Label(total_container, text="0", font=("Arial", 10, "bold"))
        self.total_label.pack(side=tk.LEFT)
        
        # Filtered (humans)
        filtered_container = ttk.Frame(stats_frame)
        filtered_container.pack(fill=tk.X, pady=5)
        ttk.Label(filtered_container, text="Filtered (Human):", width=15).pack(side=tk.LEFT)
        self.filtered_label = ttk.Label(filtered_container, text="0", font=("Arial", 10, "bold"))
        self.filtered_label.pack(side=tk.LEFT)
        
        # Status
        ttk.Separator(stats_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(stats_frame, text="Status:", font=("Arial", 9, "bold")).pack(anchor=tk.W)
        self.status_label = ttk.Label(stats_frame, text="Ready to start", foreground="gray", wraplength=250)
        self.status_label.pack(anchor=tk.W, pady=5)
        
    def setup_right_panel(self, parent):
        """Setup video display panel on the right"""
        video_frame = ttk.LabelFrame(parent, text="Detection Preview", padding=10)
        video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_canvas = tk.Canvas(video_frame, bg="black", width=1000, height=750)
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.video_canvas.create_text(
            500, 375,
            text="No input source selected\n\nChoose Camera, Video, or Image to start",
            fill="white",
            font=("Arial", 16),
            tags="placeholder"
        )
        
    def load_models(self):
        """Load YOLO models"""
        try:
            self.update_status("Loading models...")
            
            if not os.path.exists(self.model_path):
                messagebox.showerror("Model Not Found", f"Model file not found at:\n{self.model_path}")
                self.update_status("Model not found!")
                return
            
            self.model = YOLO(self.model_path)
            
            try:
                self.human_model = YOLO('yolov8n.pt')
            except:
                self.human_model = None
                messagebox.showwarning("Human Detection", "Could not load human detection model.\nHuman filter will be disabled.")
            
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = 'cuda'
                    self.model.to('cuda')
                    if self.human_model:
                        self.human_model.to('cuda')
            except:
                pass
            
            device_text = "GPU" if self.device == 'cuda' else "CPU"
            self.update_status(f"Models loaded successfully! Using {device_text}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models:\n{str(e)}")
            self.update_status("Failed to load models!")
    
    def open_camera(self):
        """Open camera for detection - FIXED VERSION"""
        if self.is_running:
            messagebox.showwarning("Already Running", "Please stop current detection first")
            return
        
        # Release existing camera
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None
            time.sleep(0.5)
        
        camera_index = self.camera_var.get()
        
        print(f"\n{'='*50}")
        print(f"Opening Camera {camera_index}...")
        print(f"{'='*50}")
        
        try:
            self.update_status(f"Opening camera {camera_index}...")
            
            # Try DirectShow first (most reliable on Windows)
            print("Trying CAP_DSHOW...")
            self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            time.sleep(1.0)  # Give more time for camera initialization
            
            if self.cap.isOpened():
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    print("✓ Camera opened with CAP_DSHOW")
                else:
                    self.cap.release()
                    self.cap = None
            
            # If CAP_DSHOW failed, try CAP_ANY
            if self.cap is None or not self.cap.isOpened():
                print("Trying CAP_ANY...")
                self.cap = cv2.VideoCapture(camera_index)
                time.sleep(1.0)
                
                if self.cap.isOpened():
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        print("✓ Camera opened with CAP_ANY")
                    else:
                        self.cap.release()
                        self.cap = None
            
            if self.cap is None or not self.cap.isOpened():
                raise Exception(
                    f"Could not open camera {camera_index}\n\n"
                    "Troubleshooting:\n"
                    "• Check if camera is connected\n"
                    "• Close other apps (Zoom, Skype, Teams)\n"
                    "• Try different camera index (0, 1, 2...)\n"
                    "• Check Windows Privacy > Camera settings\n"
                    "• Restart your computer"
                )
            
            # Configure camera
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Get actual settings
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            self.current_mode = 'camera'
            status_msg = f"Camera {camera_index} opened!\nResolution: {width}x{height}\nFPS: {fps}"
            self.update_status(status_msg)
            self.start_btn.config(state=tk.NORMAL)
            
            print(f"✓ Camera {camera_index} ready")
            print(f"  Resolution: {width}x{height}")
            print(f"  FPS: {fps}")
            print(f"{'='*50}\n")
            
        except Exception as e:
            if self.cap:
                try:
                    self.cap.release()
                except:
                    pass
                self.cap = None
            messagebox.showerror("Camera Error", str(e))
            self.update_status("Failed to open camera")
            print(f"✗ Error: {e}\n")
    
    def load_video(self):
        """Load video file"""
        if self.is_running:
            messagebox.showwarning("Already Running", "Please stop current detection first")
            return
        
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            self.cap = cv2.VideoCapture(file_path)
            
            if not self.cap.isOpened():
                raise Exception("Could not open video file")
            
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            self.current_mode = 'video'
            self.update_status(f"Video loaded: {Path(file_path).name}\nDuration: {duration:.1f}s, FPS: {fps:.1f}")
            self.start_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Video Error", f"Failed to load video:\n{str(e)}")
            self.update_status("Failed to load video")
    
    def load_image(self):
        """Load image file"""
        if self.is_running:
            messagebox.showwarning("Already Running", "Please stop current detection first")
            return
        
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            self.current_frame = cv2.imread(file_path)
            
            if self.current_frame is None:
                raise Exception("Could not load image")
            
            self.current_mode = 'image'
            self.update_status(f"Image loaded: {Path(file_path).name}")
            self.start_btn.config(state=tk.NORMAL)
            
            self.process_single_image()
            
        except Exception as e:
            messagebox.showerror("Image Error", f"Failed to load image:\n{str(e)}")
            self.update_status("Failed to load image")
    
    def process_single_image(self):
        """Process a single image"""
        if self.current_frame is None:
            return
        
        img = self.current_frame.copy()
        human_boxes = self.detect_humans(img)
        results = self.model(img, stream=True, verbose=False, device=self.device)
        img, stats = self.process_detections(img, results, human_boxes)
        
        self.recyclable_count = stats['recyclable']
        self.non_recyclable_count = stats['non_recyclable']
        self.total_detections = stats['total']
        self.filtered_count = stats['filtered']
        
        self.update_statistics()
        self.display_frame(img)
        self.update_status("Image processed successfully!")
    
    def start_detection(self):
        """Start detection process"""
        if self.model is None:
            messagebox.showerror("Error", "Models not loaded!")
            return
        
        if self.current_mode is None:
            messagebox.showwarning("No Input", "Please select an input source first")
            return
        
        if self.current_mode == 'image':
            self.process_single_image()
            return
        
        if self.cap is None or not self.cap.isOpened():
            messagebox.showerror("Error", "No valid input source")
            return
        
        self.is_running = True
        self.is_paused = False
        
        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)
        
        self.total_detections = 0
        self.recyclable_count = 0
        self.non_recyclable_count = 0
        self.filtered_count = 0
        
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()
        
        self.update_status("Detection started!")
    
    def detection_loop(self):
        """Main detection loop"""
        prev_time = time.time()
        human_detect_counter = 0
        human_boxes = []
        frame_count = 0
        
        print("\nStarting detection loop...")
        
        while self.is_running:
            try:
                if self.is_paused:
                    time.sleep(0.1)
                    continue
                
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    if self.current_mode == 'video':
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        print("Failed to read frame")
                        break
                
                frame_count += 1
                self.last_frame = frame.copy()
                
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time
                self.fps_history.append(fps)
                
                human_detect_counter += 1
                if human_detect_counter % 3 == 0:
                    human_boxes = self.detect_humans(frame)
                
                results = self.model(frame, stream=True, verbose=False, device=self.device,
                                   half=True if self.device == 'cuda' else False)
                
                frame, stats = self.process_detections(frame, results, human_boxes)
                
                self.recyclable_count = stats['recyclable']
                self.non_recyclable_count = stats['non_recyclable']
                self.total_detections += stats['total']
                self.filtered_count += stats['filtered']
                
                frame = self.draw_frame_ui(frame, fps, human_boxes)
                
                self.display_frame(frame)
                self.update_statistics()
                
                if frame_count % 30 == 0:
                    print(f"Frame {frame_count}: FPS={int(fps)}, Detections={stats['total']}")
                
                time.sleep(0.001)
                
            except Exception as e:
                print(f"Error in detection loop: {e}")
                break
        
        print("Detection loop ended")
        self.is_running = False
    
    def detect_humans(self, img):
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
                        margin = 20
                        human_boxes.append({
                            'x1': int(x1) - margin,
                            'y1': int(y1) - margin,
                            'x2': int(x2) + margin,
                            'y2': int(y2) + margin
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
        """Check if item is recyclable"""
        class_lower = class_name.lower()
        
        for recyclable in self.recyclable_items:
            if recyclable.lower() in class_lower:
                return True
        
        recyclable_keywords = ['bottle', 'can', 'paper', 'cardboard', 'glass', 
                              'metal', 'aluminium', 'aluminum']
        
        for keyword in recyclable_keywords:
            if keyword in class_lower:
                return True
        
        return False
    
    def process_detections(self, img, results, human_boxes):
        """Process detections and draw on image"""
        detection_count = 0
        recyclable_count = 0
        non_recyclable_count = 0
        filtered_count = 0
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
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
                    detection_count += 1
                    
                    class_name = self.model.names[cls] if cls < len(self.model.names) else f"Class_{cls}"
                    recyclable = self.is_recyclable(class_name)
                    
                    if recyclable:
                        recyclable_count += 1
                    else:
                        non_recyclable_count += 1
                    
                    color = (0, 255, 0) if recyclable else (0, 0, 255)
                    status = "[R]" if recyclable else "[NR]"
                    
                    if len(class_name) > 15:
                        class_name = class_name[:12] + "..."
                    
                    cvzone.cornerRect(img, (x1, y1, w, h), l=15, t=2, colorR=color)
                    cvzone.putTextRect(img, f'{status} {class_name} {conf:.2f}',
                                     (max(0, x1), max(35, y1)),
                                     scale=0.7, thickness=1,
                                     colorR=color, colorT=(255, 255, 255))
        
        stats = {
            'total': detection_count,
            'recyclable': recyclable_count,
            'non_recyclable': non_recyclable_count,
            'filtered': filtered_count
        }
        
        return img, stats
    
    def draw_frame_ui(self, img, fps, human_boxes):
        """Draw UI elements on frame"""
        height, width = img.shape[:2]
        
        for hbox in human_boxes:
            cv2.rectangle(img, (hbox['x1'], hbox['y1']), (hbox['x2'], hbox['y2']), (0, 255, 255), 2)
            cv2.putText(img, "HUMAN", (hbox['x1'], hbox['y1']-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
        
        fps_color = (0, 255, 0) if fps > 20 else (0, 255, 255) if fps > 10 else (0, 0, 255)
        cv2.putText(img, f"FPS: {int(fps)}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
        cv2.putText(img, f"Conf: {self.confidence:.2f}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        filter_status = "Filter: ON" if self.enable_human_filter else "Filter: OFF"
        cv2.putText(img, filter_status, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
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
            
            frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
            
            img_pil = Image.fromarray(frame_resized)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            
            self.video_canvas.delete("all")
            self.video_canvas.create_image(canvas_width // 2, canvas_height // 2, image=img_tk, anchor=tk.CENTER)
            self.video_canvas.image = img_tk
    
    def update_statistics(self):
        """Update statistics labels"""
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        
        self.fps_label.config(text=f"{int(avg_fps)}")
        self.recyclable_label.config(text=str(self.recyclable_count))
        self.non_recyclable_label.config(text=str(self.non_recyclable_count))
        self.total_label.config(text=str(self.total_detections))
        self.filtered_label.config(text=str(self.filtered_count))
    
    def toggle_pause(self):
        """Toggle pause state"""
        self.is_paused = not self.is_paused
        
        if self.is_paused:
            self.pause_btn.config(text="▶ Resume")
            self.update_status("Paused")
        else:
            self.pause_btn.config(text="⏸ Pause")
            self.update_status("Running...")
    
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
        
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED, text="⏸ Pause")
        self.stop_btn.config(state=tk.DISABLED)
        
        self.update_status("Detection stopped")
    
    def save_screenshot(self):
        """Save current frame as screenshot"""
        timestamp = int(time.time())
        filename = f"trash_detection_{timestamp}.jpg"
        
        if self.last_frame is not None:
            cv2.imwrite(filename, self.last_frame)
            messagebox.showinfo("Screenshot Saved", f"Saved as:\n{filename}")
        elif self.current_frame is not None:
            cv2.imwrite(filename, self.current_frame)
            messagebox.showinfo("Screenshot Saved", f"Saved as:\n{filename}")
        else:
            messagebox.showwarning("No Frame", "No frame available to save")
    
    def on_confidence_change(self, value):
        """Handle confidence slider change"""
        self.confidence = float(value)
        self.confidence_label.config(text=f"{self.confidence:.2f}")
        
        if self.current_mode == 'image' and not self.is_running:
            self.process_single_image()
    
    def toggle_human_filter(self):
        """Toggle human filter"""
        self.enable_human_filter = self.human_filter_var.get()
        status = "enabled" if self.enable_human_filter else "disabled"
        self.update_status(f"Human filter {status}")
        
        if self.current_mode == 'image' and not self.is_running:
            self.process_single_image()
    
    def update_status(self, message):
        """Update status label"""
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_running:
            if messagebox.askokcancel("Quit", "Detection is running. Stop and quit?"):
                self.stop_detection()
                self.root.destroy()
        else:
            self.root.destroy()


def main():
    root = tk.Tk()
    app = TrashDetectorGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()