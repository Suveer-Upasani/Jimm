# import cv2
# import mediapipe as mp
# import numpy as np
# import time
# import math
# import csv
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_agg import FigureCanvasAgg
# from collections import deque
# import pandas as pd
# from datetime import datetime

# # -------------------- Initialize MediaPipe --------------------
# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils
# mp_selfie_segmentation = mp.solutions.selfie_segmentation

# pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
# segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# # -------------------- Camera Setup --------------------
# working_camera = None
# for index in range(10):
#     temp_cap = cv2.VideoCapture(index)
#     time.sleep(0.3)
#     if temp_cap.isOpened():
#         ret, frame = temp_cap.read()
#         if ret and frame is not None:
#             working_camera = index
#             temp_cap.release()
#             break
#     temp_cap.release()

# if working_camera is None:
#     print("❌ No working camera found.")
#     exit(1)

# cap = cv2.VideoCapture(working_camera)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# # -------------------- CSV Setup --------------------
# csv_file = "deadlift_reps_log.csv"
# csv_fields = [
#     "rep", "start_time", "end_time", "duration_sec", "side",
#     "avg_torso_angle", "min_torso_angle", "max_torso_angle",
#     "avg_hip_angle", "min_hip_angle", "max_hip_angle",
#     "avg_knee_angle", "min_knee_angle", "max_knee_angle",
#     "avg_hip_alignment", "min_hip_alignment", "hip_displacement",
#     "rom_quality", "rep_speed", "back_angle_quality", "hip_knee_alignment",
#     "peak_velocity", "mean_velocity", "sticking_point_velocity",
#     "spinal_compression", "hip_moment", "knee_shear",
#     "work_efficiency", "technical_consistency", "fatigue_index"
# ]
# rep_data = []

# # -------------------- Advanced Biomechanical Analysis --------------------
# class BiomechanicalAnalyzer:
#     def __init__(self):
#         self.body_weight = 75  # kg - default, can be user configured
#         self.barbell_weight = 60  # kg - default
        
#     def calculate_spinal_loading(self, landmarks, width, height):
#         """Estimate spinal compression forces"""
#         hip_px = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * width),
#                   int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * height))
#         shoulder_px = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width),
#                        int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height))
        
#         # Calculate back angle
#         back_angle = math.atan2(shoulder_px[0]-hip_px[0], shoulder_px[1]-hip_px[1])
#         back_angle_deg = math.degrees(back_angle)
        
#         # Simplified spinal compression calculation
#         compression_force = self.barbell_weight * (1 + abs(math.sin(back_angle)))
#         return compression_force
    
#     def calculate_hip_moment(self, landmarks, width, height):
#         """Calculate torque around hip joint"""
#         hip_px = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * width),
#                   int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * height))
#         shoulder_px = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width),
#                        int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height))
        
#         # Torso length estimation
#         torso_length = math.hypot(shoulder_px[0]-hip_px[0], shoulder_px[1]-hip_px[1])
        
#         # Hip moment increases with forward lean
#         moment_arm = torso_length * abs(math.sin(math.radians(self.calculate_torso_angle(landmarks, width, height))))
#         hip_moment = self.body_weight * moment_arm
#         return hip_moment
    
#     def calculate_knee_shear(self, landmarks, width, height):
#         """Estimate anterior shear forces on knees"""
#         knee_px = (int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * width),
#                    int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * height))
#         ankle_px = (int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * width),
#                     int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * height))
        
#         # Shank angle
#         shank_angle = math.atan2(knee_px[0]-ankle_px[0], knee_px[1]-ankle_px[1])
        
#         # Shear force increases with horizontal shank position
#         shear_force = self.barbell_weight * abs(math.cos(shank_angle))
#         return shear_force
    
#     def calculate_torso_angle(self, landmarks, width, height):
#         """Calculate torso angle with vertical"""
#         hip_px = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * width),
#                   int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * height))
#         shoulder_px = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width),
#                        int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height))
        
#         return angle_with_vertical(shoulder_px[0]-hip_px[0], shoulder_px[1]-hip_px[1])

# # -------------------- Velocity Analysis --------------------
# class VelocityAnalyzer:
#     def __init__(self):
#         self.velocity_thresholds = {
#             'explosive': 1.0,  # m/s
#             'moderate': 0.5,   # m/s  
#             'slow': 0.3,       # m/s
#             'grinding': 0.1    # m/s
#         }
#         self.bar_path_history = deque(maxlen=100)
#         self.timestamps = deque(maxlen=100)
    
#     def analyze_velocity(self, bar_position, timestamp):
#         """Analyze bar velocity throughout movement"""
#         self.bar_path_history.append(bar_position)
#         self.timestamps.append(timestamp)
        
#         if len(self.bar_path_history) < 2:
#             return None
            
#         # Calculate instantaneous velocity
#         current_pos = np.array(self.bar_path_history[-1])
#         prev_pos = np.array(self.bar_path_history[-2])
#         time_diff = self.timestamps[-1] - self.timestamps[-2]
        
#         if time_diff <= 0:
#             return None
            
#         displacement = np.linalg.norm(current_pos - prev_pos)
#         velocity = displacement / time_diff
        
#         return velocity
    
#     def get_velocity_profile(self):
#         """Get complete velocity profile"""
#         if len(self.bar_path_history) < 2:
#             return None
            
#         velocities = []
#         for i in range(1, len(self.bar_path_history)):
#             current_pos = np.array(self.bar_path_history[i])
#             prev_pos = np.array(self.bar_path_history[i-1])
#             time_diff = self.timestamps[i] - self.timestamps[i-1]
            
#             if time_diff > 0:
#                 displacement = np.linalg.norm(current_pos - prev_pos)
#                 velocities.append(displacement / time_diff)
        
#         if not velocities:
#             return None
            
#         return {
#             'current_velocity': velocities[-1] if velocities else 0,
#             'peak_velocity': max(velocities) if velocities else 0,
#             'mean_velocity': np.mean(velocities) if velocities else 0,
#             'velocity_profile': velocities
#         }

# # -------------------- Fault Detection --------------------
# class FaultDetector:
#     def __init__(self):
#         self.fault_thresholds = {
#             'spinal_flexion': 15,  # degrees from neutral
#             'hip_rise_lead': 0.2,  # seconds
#             'knee_valgus': 10,     # degrees from vertical
#             'asymmetry': 5,        # percentage difference
#         }
    
#     def detect_rounded_back(self, spine_angle, phase):
#         """Detect thoracic and lumbar flexion"""
#         faults = []
        
#         if phase == 'descent' and spine_angle > self.fault_thresholds['spinal_flexion']:
#             faults.append({
#                 'type': 'LUMBAR_FLEXION',
#                 'severity': spine_angle / self.fault_thresholds['spinal_flexion'],
#                 'phase': phase,
#                 'correction': 'Brace core and maintain neutral spine'
#             })
        
#         return faults
    
#     def detect_knee_valgus(self, knee_px, ankle_px):
#         """Detect knee cave (valgus collapse)"""
#         # Calculate angle from vertical
#         angle = abs(math.degrees(math.atan2(knee_px[0] - ankle_px[0], knee_px[1] - ankle_px[1])))
        
#         if angle > self.fault_thresholds['knee_valgus']:
#             return {
#                 'type': 'KNEE_VALGUS',
#                 'severity': angle / self.fault_thresholds['knee_valgus'],
#                 'max_angle': angle,
#                 'correction': 'Drive knees out, engage glutes'
#             }
#         return None

# # -------------------- Enhanced Helper Functions --------------------
# def lm_to_pixel(landmark, width, height):
#     return int(landmark.x * width), int(landmark.y * height)

# def angle_with_vertical(vx, vy):
#     # Vector for vertical (0, -1)
#     dot = vx * 0 + vy * (-1)
#     mag_v = math.hypot(vx, vy)
#     if mag_v == 0:
#         return 0.0
#     # Normalize dot product and clamp to [-1, 1] for safe acos
#     cos_a = max(min(dot / mag_v, 1), -1)
#     return math.degrees(math.acos(cos_a))

# def angle_between_points(a, b, c):
#     ba = np.array([a[0]-b[0], a[1]-b[1]])
#     bc = np.array([c[0]-b[0], c[1]-b[1]])
#     # Add a small epsilon to prevent division by zero
#     cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
#     cos_angle = max(min(cos_angle,1),-1)
#     return math.degrees(np.arccos(cos_angle))

# def calculate_deadlift_reference_points(ankle_px, knee_px, shoulder_px, hip_px, frame_height):
#     """
#     Calculate reference points for proper deadlift form using plus sign system
#     """
#     # Reference 1: Vertical line from mid-foot (critical for bar path)
#     mid_foot_x = ankle_px[0]
    
#     # Reference 2: Horizontal line for optimal hip height
#     knee_height = knee_px[1]
#     shoulder_height = shoulder_px[1]
#     optimal_hip_height = knee_height + (shoulder_height - knee_height) * 0.6
    
#     # Reference 3: Back angle reference line
#     back_angle_ref_y = ankle_px[1] - frame_height * 0.3
    
#     return {
#         'vertical_line_x': mid_foot_x,
#         'optimal_hip_height': optimal_hip_height,
#         'back_angle_ref': back_angle_ref_y,
#         'mid_foot': (mid_foot_x, ankle_px[1])
#     }

# def draw_plus_sign_reference(img, reference_points, size=150, color=(0, 255, 255), thickness=3):
#     """
#     Draw enhanced plus sign reference system for deadlift form
#     """
#     vx = reference_points['vertical_line_x']
#     hip_y = int(reference_points['optimal_hip_height'])
#     mid_foot = reference_points['mid_foot']
    
#     # Vertical reference line (bar path should follow this)
#     cv2.line(img, (vx, 0), (vx, img.shape[0]), color, thickness-1)
    
#     # Horizontal reference line (optimal hip height)
#     cv2.line(img, (vx-size, hip_y), (vx+size, hip_y), color, thickness-1)
    
#     # Plus sign center
#     center_x, center_y = vx, hip_y
    
#     # Enhanced plus sign with crosshair
#     arm_length = size // 2
#     # Vertical arm
#     cv2.line(img, (center_x, center_y - arm_length), (center_x, center_y + arm_length), 
#              (255, 255, 255), thickness+1)
#     # Horizontal arm  
#     cv2.line(img, (center_x - arm_length, center_y), (center_x + arm_length, center_y), 
#              (255, 255, 255), thickness+1)
    
#     # Outer circle
#     cv2.circle(img, (center_x, center_y), arm_length, color, thickness)
    
#     # Inner targeting circle
#     cv2.circle(img, (center_x, center_y), 15, (0, 0, 255), -1)
    
#     # Labels
#     cv2.putText(img, "IDEAL HIP HEIGHT", (center_x + 40, center_y - 10), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
#     cv2.putText(img, "BAR PATH", (center_x + 10, 30), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
#     # Draw mid-foot marker
#     cv2.circle(img, mid_foot, 8, (0, 255, 0), -1)
#     cv2.putText(img, "MID-FOOT", (mid_foot[0] - 40, mid_foot[1] + 25), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

# def calculate_form_quality(hip_px, knee_px, shoulder_px, ankle_px, reference_points):
#     """
#     Calculate comprehensive form quality metrics using reference system
#     """
#     vx = reference_points['vertical_line_x']
#     optimal_hip_y = reference_points['optimal_hip_height']
    
#     # 1. Hip alignment quality (vertical reference)
#     hip_v_alignment = 100 - min(100, (abs(hip_px[0] - vx) / (vx * 0.2 + 1e-6)) * 100)
    
#     # 2. Hip height quality (horizontal reference)
#     hip_h_alignment = 100 - min(100, (abs(hip_px[1] - optimal_hip_y) / (optimal_hip_y * 0.3 + 1e-6)) * 100)
    
#     # 3. Back angle quality (should be ~45-60 degrees at setup)
#     back_angle = angle_with_vertical(shoulder_px[0]-hip_px[0], shoulder_px[1]-hip_px[1])
#     back_quality = 100 - min(100, abs(back_angle - 50) / 50 * 100)
    
#     # 4. Hip-knee-ankle alignment (proportions)
#     hip_knee_dist = math.hypot(hip_px[0]-knee_px[0], hip_px[1]-knee_px[1])
#     knee_ankle_dist = math.hypot(knee_px[0]-ankle_px[0], knee_px[1]-ankle_px[1])
#     alignment_ratio = hip_knee_dist / (knee_ankle_dist + 1e-6)
#     alignment_quality = 100 - min(100, abs(alignment_ratio - 1.2) / 1.2 * 100)
    
#     overall_quality = (hip_v_alignment + hip_h_alignment + back_quality + alignment_quality) / 4
    
#     return {
#         'overall': overall_quality,
#         'hip_vertical': hip_v_alignment,
#         'hip_height': hip_h_alignment,
#         'back_angle': back_quality,
#         'hip_knee_alignment': alignment_quality
#     }

# def draw_form_feedback(img, hip_px, form_quality, reference_points):
#     """
#     Draw real-time form feedback based on reference system
#     """
#     vx = reference_points['vertical_line_x']
#     optimal_y = int(reference_points['optimal_hip_height'])
    
#     # Draw current hip position relative to reference
#     cv2.circle(img, hip_px, 12, (0, 0, 255), -1)
#     cv2.circle(img, hip_px, 12, (255, 255, 255), 2)
    
#     # Draw connection lines to reference
#     cv2.line(img, hip_px, (vx, hip_px[1]), (255, 100, 100), 2)  # Horizontal connection
#     cv2.line(img, hip_px, (hip_px[0], optimal_y), (255, 100, 100), 2)  # Vertical connection
    
#     # Quality indicator color coding
#     if form_quality['overall'] > 80:
#         quality_color = (0, 255, 0)
#     elif form_quality['overall'] > 60:
#         quality_color = (0, 255, 255)
#     else:
#         quality_color = (0, 0, 255)
    
#     # Draw quality bar
#     bar_x, bar_y = 20, img.shape[0] - 100
#     bar_width = 200
#     bar_height = 20
#     cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
#     cv2.rectangle(img, (bar_x, bar_y), (bar_x + int(bar_width * form_quality['overall'] / 100), bar_y + bar_height), quality_color, -1)
#     cv2.putText(img, f"FORM: {form_quality['overall']:.0f}%", (bar_x, bar_y - 10), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, quality_color, 2)

# def draw_trajectory_analysis(img, trajectories, current_landmarks, width, height):
#     """
#     Draw real-time trajectory analysis with lines and paths
#     """
#     # Define colors for different trajectories
#     colors = {
#         'hip': (0, 255, 255),    # Cyan
#         'shoulder': (255, 255, 0), # Yellow
#         'knee': (255, 0, 255),   # Magenta
#         'ankle': (0, 255, 0)     # Green
#     }
    
#     # Draw trajectories
#     for landmark_name, points in trajectories.items():
#         if len(points) > 1:
#             color = colors.get(landmark_name, (255, 255, 255))
#             # Draw trajectory line
#             for i in range(1, len(points)):
#                 cv2.line(img, points[i-1], points[i], color, 2)
#             # Draw current position
#             if points:
#                 cv2.circle(img, points[-1], 8, color, -1)
    
#     # Draw current landmark connections
#     if current_landmarks:
#         # Hip to knee
#         cv2.line(img, current_landmarks['hip'], current_landmarks['knee'], (0, 200, 255), 3)
#         # Knee to ankle
#         cv2.line(img, current_landmarks['knee'], current_landmarks['ankle'], (0, 200, 255), 3)
#         # Hip to shoulder
#         cv2.line(img, current_landmarks['hip'], current_landmarks['shoulder'], (255, 200, 0), 3)
        
#         # Label landmarks
#         for landmark_name, point in current_landmarks.items():
#             color = colors.get(landmark_name, (255, 255, 255))
#             cv2.putText(img, landmark_name.upper(), (point[0] + 10, point[1]), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# def create_live_graphs(angle_history, velocity_history, biomech_history):
#     """
#     Create live matplotlib graphs for analysis
#     """
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
#     fig.suptitle('Deadlift Analysis - Live Metrics', fontsize=14, fontweight='bold')
    
#     # Plot 1: Joint Angles
#     if angle_history:
#         frames = list(range(len(angle_history)))
#         ax1.clear()
#         ax1.plot(frames, [a['torso'] for a in angle_history], 'r-', label='Torso', linewidth=2)
#         ax1.plot(frames, [a['hip'] for a in angle_history], 'g-', label='Hip', linewidth=2)
#         ax1.plot(frames, [a['knee'] for a in angle_history], 'b-', label='Knee', linewidth=2)
#         ax1.set_title('Joint Angles Over Time')
#         ax1.set_xlabel('Frame')
#         ax1.set_ylabel('Angle (degrees)')
#         ax1.legend()
#         ax1.grid(True, alpha=0.3)
    
#     # Plot 2: Velocity Profile
#     if velocity_history:
#         frames = list(range(len(velocity_history)))
#         ax2.clear()
#         ax2.plot(frames, velocity_history, 'purple', linewidth=2)
#         ax2.set_title('Bar Velocity Profile')
#         ax2.set_xlabel('Frame')
#         ax2.set_ylabel('Velocity (pixels/frame)')
#         ax2.grid(True, alpha=0.3)
    
#     # Plot 3: Biomechanical Loads
#     if biomech_history:
#         frames = list(range(len(biomech_history)))
#         ax3.clear()
#         ax3.plot(frames, [b['spinal'] for b in biomech_history], 'red', label='Spinal Load', linewidth=2)
#         ax3.plot(frames, [b['hip'] for b in biomech_history], 'blue', label='Hip Moment', linewidth=2)
#         ax3.plot(frames, [b['knee'] for b in biomech_history], 'green', label='Knee Shear', linewidth=2)
#         ax3.set_title('Biomechanical Loads')
#         ax3.set_xlabel('Frame')
#         ax3.set_ylabel('Relative Load')
#         ax3.legend()
#         ax3.grid(True, alpha=0.3)
    
#     # Plot 4: Form Quality
#     if angle_history and 'form_quality' in angle_history[0]:
#         frames = list(range(len(angle_history)))
#         ax4.clear()
#         ax4.plot(frames, [a['form_quality'] for a in angle_history], 'orange', linewidth=2)
#         ax4.set_title('Form Quality Over Time')
#         ax4.set_xlabel('Frame')
#         ax4.set_ylabel('Quality Score (%)')
#         ax4.set_ylim(0, 100)
#         ax4.grid(True, alpha=0.3)
    
#     plt.tight_layout()
    
#     # Convert matplotlib figure to OpenCV image
#     canvas = FigureCanvasAgg(fig)
#     canvas.draw()
#     buf = canvas.buffer_rgba()
#     graph_img = np.asarray(buf)
#     graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGBA2BGR)
    
#     plt.close(fig)
#     return graph_img

# # -------------------- Classes and Rep Detection --------------------
# class Button:
#     def __init__(self, x, y, w, h, text, color, text_color):
#         self.x = x
#         self.y = y
#         self.w = w
#         self.h = h
#         self.text = text
#         self.color = color
#         self.text_color = text_color
#         self.hovered = False
    
#     def draw(self, img):
#         # Slightly brighten color on hover
#         color = tuple(int(c * 1.2) if c * 1.2 <= 255 else 255 for c in self.color) if self.hovered else self.color
#         cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), color, -1)
#         cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (255, 255, 255), 2)
        
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 0.5
#         thickness = 2
#         text_size = cv2.getTextSize(self.text, font, font_scale, thickness)[0]
#         text_x = self.x + (self.w - text_size[0]) // 2
#         text_y = self.y + (self.h + text_size[1]) // 2
#         cv2.putText(img, self.text, (text_x, text_y), font, font_scale, self.text_color, thickness)
    
#     def is_clicked(self, x, y):
#         return self.x <= x <= self.x + self.w and self.y <= y <= self.y + self.h

# class DeadliftRepDetector:
#     def __init__(self):
#         self.state = "STANDING"
#         self.rep_count = 0
#         self.hip_trajectory = deque(maxlen=100)
#         self.current_rep_metrics = []
#         self.rep_start_time = None
#         # Angle thresholds are for a side-view deadlift where 180 is straight up (STANDING)
#         self.standing_threshold = 160
#         self.bottom_threshold = 80
#         # Normalised Y-coordinate change (0 to 1)
#         self.hip_height_threshold = 0.005 # Reduced sensitivity
#         self.last_hip_y = None
        
#     def update(self, hip_y_norm, torso_angle, hip_angle, knee_angle, form_quality):
#         rep_completed = False
        
#         if self.last_hip_y is None:
#             self.last_hip_y = hip_y_norm
#             return rep_completed, self.state
        
#         # Hip Y-coordinate increases as the hip moves down (MediaPipe convention)
#         hip_moving_down = hip_y_norm > self.last_hip_y + self.hip_height_threshold
#         hip_moving_up = hip_y_norm < self.last_hip_y - self.hip_height_threshold
        
#         if self.state == "STANDING":
#             # Start of descent: Torso bends AND hip is moving down
#             if torso_angle < self.standing_threshold - 10 and hip_moving_down:
#                 self.state = "DESCENDING"
#                 self.rep_start_time = time.time()
#                 self.current_rep_metrics = []
#                 print(f"  → Rep {self.rep_count + 1} started (Descending)")
                
#         elif self.state == "DESCENDING":
#             # Reached bottom: Torso is bent low AND hip angle is small
#             if torso_angle < self.bottom_threshold and hip_angle < 110:
#                 self.state = "BOTTOM"
#                 print(f"  → Bottom position reached")
                
#         elif self.state == "BOTTOM":
#             # Start of ascent: Hip is moving up AND torso is extending
#             if hip_moving_up and torso_angle > self.bottom_threshold + 5:
#                 self.state = "ASCENDING"
#                 print(f"  → Ascending")
                
#         elif self.state == "ASCENDING":
#             # Rep complete: Torso is upright AND hip is fully extended
#             if torso_angle > self.standing_threshold - 5 and hip_angle > 165:
#                 self.state = "STANDING"
#                 self.rep_count += 1
#                 rep_completed = True
#                 print(f"✓ Rep {self.rep_count} COMPLETED!")
        
#         self.last_hip_y = hip_y_norm
#         return rep_completed, self.state
    
#     def add_metrics(self, torso, hip, knee, form_metrics, biomech_metrics, velocity_metrics):
#         self.current_rep_metrics.append([torso, hip, knee, form_metrics, biomech_metrics, velocity_metrics])
    
#     def get_rep_summary(self):
#         if not self.current_rep_metrics:
#             return None
        
#         metrics = np.array([m[:3] for m in self.current_rep_metrics])
#         form_metrics = [m[3] for m in self.current_rep_metrics]
#         biomech_metrics = [m[4] for m in self.current_rep_metrics]
#         velocity_metrics = [m[5] for m in self.current_rep_metrics]
        
#         duration = time.time() - self.rep_start_time if self.rep_start_time else 0
        
#         avg_form_quality = np.mean([fm['overall'] for fm in form_metrics])
#         avg_back_quality = np.mean([fm['back_angle'] for fm in form_metrics])
#         avg_alignment_quality = np.mean([fm['hip_knee_alignment'] for fm in form_metrics])
        
#         # Biomechanical averages
#         avg_spinal_load = np.mean([bm['spinal'] for bm in biomech_metrics])
#         avg_hip_moment = np.mean([bm['hip'] for bm in biomech_metrics])
#         avg_knee_shear = np.mean([bm['knee'] for bm in biomech_metrics])
        
#         # Velocity analysis
#         peak_velocity = max([vm.get('current_velocity', 0) for vm in velocity_metrics if vm]) if any(velocity_metrics) else 0
#         mean_velocity = np.mean([vm.get('current_velocity', 0) for vm in velocity_metrics if vm]) if any(velocity_metrics) else 0
        
#         return {
#             'torso': (np.mean(metrics[:, 0]), np.min(metrics[:, 0]), np.max(metrics[:, 0])),
#             'hip': (np.mean(metrics[:, 1]), np.min(metrics[:, 1]), np.max(metrics[:, 1])),
#             'knee': (np.mean(metrics[:, 2]), np.min(metrics[:, 2]), np.max(metrics[:, 2])),
#             'form_quality': avg_form_quality,
#             'back_quality': avg_back_quality,
#             'alignment_quality': avg_alignment_quality,
#             'duration': duration,
#             'rom_quality': np.max(metrics[:, 0]) - np.min(metrics[:, 0]),
#             'spinal_load': avg_spinal_load,
#             'hip_moment': avg_hip_moment,
#             'knee_shear': avg_knee_shear,
#             'peak_velocity': peak_velocity,
#             'mean_velocity': mean_velocity
#         }

# # -------------------- Initialize Systems --------------------
# rep_detector = DeadliftRepDetector()
# biomech_analyzer = BiomechanicalAnalyzer()
# velocity_analyzer = VelocityAnalyzer()
# fault_detector = FaultDetector()

# # Trajectory tracking
# trajectories = {
#     'hip': deque(maxlen=50),
#     'shoulder': deque(maxlen=50),
#     'knee': deque(maxlen=50),
#     'ankle': deque(maxlen=50)
# }

# # History for graphs
# angle_history = deque(maxlen=100)
# velocity_history = deque(maxlen=100)
# biomech_history = deque(maxlen=100)

# hip_trajectory = deque(maxlen=150)
# blur_enabled = True
# logging_active = False
# show_reference = True
# show_trajectory = True
# show_graphs = True

# # Create buttons
# button_y = 10
# button_spacing = 10
# button_width = 100
# button_height = 40

# start_btn = Button(10, button_y, button_width, button_height, "START", (0, 200, 0), (255, 255, 255))
# stop_btn = Button(10 + button_width + button_spacing, button_y, button_width, button_height, "STOP", (0, 0, 200), (255, 255, 255))
# blur_btn = Button(10 + 2*(button_width + button_spacing), button_y, button_width, button_height, "BLUR ON", (200, 100, 0), (255, 255, 255))
# ref_btn = Button(10 + 3*(button_width + button_spacing), button_y, button_width, button_height, "REF ON", (100, 100, 200), (255, 255, 255))
# traj_btn = Button(10 + 4*(button_width + button_spacing), button_y, button_width, button_height, "TRAJ ON", (150, 150, 0), (255, 255, 255))
# graph_btn = Button(10 + 5*(button_width + button_spacing), button_y, button_width, button_height, "GRAPHS ON", (200, 100, 200), (255, 255, 255))
# quit_btn = Button(10 + 6*(button_width + button_spacing), button_y, button_width, button_height, "QUIT", (200, 0, 0), (255, 255, 255))

# buttons = [start_btn, stop_btn, blur_btn, ref_btn, traj_btn, graph_btn, quit_btn]

# # -------------------- Mouse Callback --------------------
# mouse_x, mouse_y = 0, 0
# mouse_clicked = False

# def mouse_callback(event, x, y, flags, param):
#     """Handles mouse events for button clicks and hover."""
#     global mouse_x, mouse_y, mouse_clicked
#     mouse_x, mouse_y = x, y
#     if event == cv2.EVENT_LBUTTONDOWN:
#         mouse_clicked = True

# # -------------------- Window Initialization --------------------
# cv2.namedWindow("Deadlift Form Tracker")
# cv2.setMouseCallback("Deadlift Form Tracker", mouse_callback)

# print("\n" + "="*60)
# print("ADVANCED DEADLIFT FORM TRACKER - Professional Biomechanical Analysis")
# print("="*60)
# print("Features:")
# print("  ⊕ Real-time trajectory analysis with colored paths")
# print("  ⊕ Live biomechanical load monitoring")
# print("  ⊕ Velocity-based performance analysis") 
# print("  ⊕ Comprehensive graphing and CSV logging")
# print("="*60)

# # -------------------- Main Loop --------------------
# while True:
#     ret, frame = cap.read()
#     if not ret or frame is None:
#         if working_camera is not None:
#             print("⚠️ Warning: Camera frame retrieval failed. Exiting...")
#         break

#     frame = cv2.flip(frame, 1)
#     h, w, _ = frame.shape
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     if blur_enabled:
#         # Selfie segmentation for background blur
#         seg_results = segmentation.process(rgb_frame)
#         mask = seg_results.segmentation_mask
#         condition = np.stack((mask,)*3, axis=-1) > 0.5
#         blurred_frame = cv2.GaussianBlur(frame, (55,55), 0)
#         frame = np.where(condition, frame, blurred_frame)

#     # Black canvas for skeleton visualization
#     black_canvas = np.zeros((h, w, 3), dtype=np.uint8)

#     pose_results = pose.process(rgb_frame)
    
#     current_landmarks = None
#     current_timestamp = time.time()
    
#     if pose_results.pose_landmarks:
#         lm = pose_results.pose_landmarks.landmark
        
#         # Determine which side (left/right) is more visible for side-view tracking
#         left_hip_vis = lm[mp_pose.PoseLandmark.LEFT_HIP].visibility
#         right_hip_vis = lm[mp_pose.PoseLandmark.RIGHT_HIP].visibility
#         use_left = left_hip_vis >= right_hip_vis and left_hip_vis > 0.5
#         use_right = right_hip_vis > left_hip_vis and right_hip_vis > 0.5
        
#         if use_left or use_right:
#             side_name = "LEFT" if use_left else "RIGHT"
            
#             if use_left:
#                 hip_idx, knee_idx, ankle_idx, shoulder_idx = (
#                     mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE,
#                     mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_SHOULDER
#                 )
#             else:
#                 hip_idx, knee_idx, ankle_idx, shoulder_idx = (
#                     mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE,
#                     mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_SHOULDER
#                 )

#             # Get pixel coordinates
#             hip_px = lm_to_pixel(lm[hip_idx], w, h)
#             knee_px = lm_to_pixel(lm[knee_idx], w, h)
#             ankle_px = lm_to_pixel(lm[ankle_idx], w, h)
#             shoulder_px = lm_to_pixel(lm[shoulder_idx], w, h)

#             # Update trajectories
#             trajectories['hip'].append(hip_px)
#             trajectories['shoulder'].append(shoulder_px)
#             trajectories['knee'].append(knee_px)
#             trajectories['ankle'].append(ankle_px)
            
#             current_landmarks = {
#                 'hip': hip_px,
#                 'shoulder': shoulder_px,
#                 'knee': knee_px,
#                 'ankle': ankle_px
#             }

#             # Calculate enhanced reference points
#             reference_points = calculate_deadlift_reference_points(
#                 ankle_px, knee_px, shoulder_px, hip_px, h
#             )
            
#             # Calculate comprehensive form quality
#             form_quality = calculate_form_quality(
#                 hip_px, knee_px, shoulder_px, ankle_px, reference_points
#             )

#             # Calculate angles
#             torso_angle = angle_with_vertical(shoulder_px[0]-hip_px[0], shoulder_px[1]-hip_px[1])
#             hip_angle = angle_between_points(shoulder_px, hip_px, knee_px)
#             knee_angle = angle_between_points(hip_px, knee_px, ankle_px)

#             # Biomechanical analysis
#             spinal_load = biomech_analyzer.calculate_spinal_loading(lm, w, h)
#             hip_moment = biomech_analyzer.calculate_hip_moment(lm, w, h)
#             knee_shear = biomech_analyzer.calculate_knee_shear(lm, w, h)
            
#             biomech_metrics = {
#                 'spinal': spinal_load,
#                 'hip': hip_moment,
#                 'knee': knee_shear
#             }

#             # Velocity analysis (using hip as proxy for bar path)
#             velocity_metrics = velocity_analyzer.analyze_velocity(hip_px, current_timestamp)
#             velocity_profile = velocity_analyzer.get_velocity_profile()
            
#             # Update history for graphs
#             angle_history.append({
#                 'torso': torso_angle,
#                 'hip': hip_angle,
#                 'knee': knee_angle,
#                 'form_quality': form_quality['overall']
#             })
            
#             if velocity_profile:
#                 velocity_history.append(velocity_profile['current_velocity'])
#             else:
#                 velocity_history.append(0)
                
#             biomech_history.append(biomech_metrics)

#             # Track hip trajectory
#             hip_trajectory.append(hip_px)

#             # Rep detection and logging
#             if logging_active:
#                 # Add current frame's metrics
#                 rep_detector.add_metrics(torso_angle, hip_angle, knee_angle, form_quality, biomech_metrics, velocity_profile)
                
#                 # Check for rep state change
#                 rep_completed, current_state = rep_detector.update(
#                     lm[hip_idx].y, torso_angle, hip_angle, knee_angle, form_quality
#                 )
                
#                 if rep_completed:
#                     summary = rep_detector.get_rep_summary()
#                     if summary:
#                         # Append the full set of logged data for the completed rep
#                         rep_data.append([
#                             rep_detector.rep_count,
#                             rep_detector.rep_start_time,
#                             time.time(),
#                             summary['duration'],
#                             side_name,
#                             summary['torso'][0], summary['torso'][1], summary['torso'][2],
#                             summary['hip'][0], summary['hip'][1], summary['hip'][2],
#                             summary['knee'][0], summary['knee'][1], summary['knee'][2],
#                             form_quality['hip_vertical'], form_quality['hip_height'],
#                             abs(hip_px[0] - reference_points['vertical_line_x']),
#                             summary['rom_quality'],
#                             1.0 / summary['duration'] if summary['duration'] > 0 else 0,
#                             summary['back_quality'],
#                             summary['alignment_quality'],
#                             summary['peak_velocity'],
#                             summary['mean_velocity'],
#                             0,  # sticking_point_velocity placeholder
#                             summary['spinal_load'],
#                             summary['hip_moment'], 
#                             summary['knee_shear'],
#                             85.0,  # work_efficiency placeholder
#                             90.0,  # technical_consistency placeholder
#                             0.1    # fatigue_index placeholder
#                         ])

#             # Draw skeleton
#             for canvas in [frame, black_canvas]:
#                 # Draw connections and landmarks
#                 mp_drawing.draw_landmarks(
#                     canvas, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                     landmark_drawing_spec=mp_drawing.DrawingSpec(color=(40,40,40), thickness=8, circle_radius=6),
#                     connection_drawing_spec=mp_drawing.DrawingSpec(color=(60,60,60), thickness=12)
#                 )
#                 mp_drawing.draw_landmarks(
#                     canvas, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                     landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
#                     connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,200,255), thickness=3)
#                 )

#             # Draw enhanced reference system
#             if show_reference:
#                 for canvas in [frame, black_canvas]:
#                     draw_plus_sign_reference(canvas, reference_points, size=200)
#                     draw_form_feedback(canvas, hip_px, form_quality, reference_points)

#             # Draw trajectory analysis
#             if show_trajectory:
#                 for canvas in [frame, black_canvas]:
#                     draw_trajectory_analysis(canvas, trajectories, current_landmarks, w, h)

#             # Enhanced display info
#             status_color = (0, 255, 0) if logging_active else (100, 100, 100)
#             cv2.putText(frame, f"Reps: {rep_detector.rep_count} | State: {rep_detector.state}", 
#                         (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
#             cv2.putText(frame, f"Form Quality: {form_quality['overall']:.0f}% | Back: {form_quality['back_angle']:.0f}%", 
#                         (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
#             cv2.putText(frame, f"Hip Alignment: V{form_quality['hip_vertical']:.0f}% H{form_quality['hip_height']:.0f}%", 
#                         (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
            
#             # Biomechanical data display
#             cv2.putText(frame, f"Spinal Load: {spinal_load:.1f}N | Hip Moment: {hip_moment:.1f}Nm", 
#                         (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
            
#             if velocity_profile:
#                 cv2.putText(frame, f"Velocity: {velocity_profile['current_velocity']:.2f} px/frame | Peak: {velocity_profile['peak_velocity']:.2f}", 
#                             (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 255), 1)
#         else:
#             cv2.putText(frame, "ADJUST VIEW: Side-view body not clearly visible.", 
#                         (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

#     # Create and display graphs
#     if show_graphs and (angle_history or velocity_history or biomech_history):
#         graph_img = create_live_graphs(list(angle_history), list(velocity_history), list(biomech_history))
#         graph_h, graph_w = graph_img.shape[:2]
        
#         # Resize graph to fit alongside video
#         graph_scale = h / graph_h * 0.8
#         new_graph_w = int(graph_w * graph_scale)
#         new_graph_h = int(graph_h * graph_scale)
#         graph_img = cv2.resize(graph_img, (new_graph_w, new_graph_h))
        
#         # Place graph to the right of the combined video
#         combined = np.hstack((frame, black_canvas))
#         if combined.shape[0] > new_graph_h:
#             # Pad graph to match height
#             padding = np.zeros((combined.shape[0] - new_graph_h, new_graph_w, 3), dtype=np.uint8)
#             graph_img = np.vstack((graph_img, padding))
        
#         combined_with_graph = np.hstack((combined, graph_img))
#     else:
#         combined_with_graph = np.hstack((frame, black_canvas))

#     # Draw buttons and handle clicks
#     for btn in buttons:
#         btn.hovered = btn.is_clicked(mouse_x, mouse_y)
#         btn.draw(combined_with_graph)

#     # Update dynamic button text
#     blur_btn.text = "BLUR ON" if blur_enabled else "BLUR OFF"
#     ref_btn.text = "REF ON" if show_reference else "REF OFF"
#     traj_btn.text = "TRAJ ON" if show_trajectory else "TRAJ OFF"
#     graph_btn.text = "GRAPHS ON" if show_graphs else "GRAPHS OFF"

#     if mouse_clicked:
#         if start_btn.is_clicked(mouse_x, mouse_y):
#             if not logging_active:
#                 logging_active = True
#                 print("✓ Tracking started...")
#         elif stop_btn.is_clicked(mouse_x, mouse_y):
#             if logging_active:
#                 logging_active = False
#                 print("✓ Tracking stopped...")
#         elif blur_btn.is_clicked(mouse_x, mouse_y):
#             blur_enabled = not blur_enabled
#         elif ref_btn.is_clicked(mouse_x, mouse_y):
#             show_reference = not show_reference
#         elif traj_btn.is_clicked(mouse_x, mouse_y):
#             show_trajectory = not show_trajectory
#         elif graph_btn.is_clicked(mouse_x, mouse_y):
#             show_graphs = not show_graphs
#         elif quit_btn.is_clicked(mouse_x, mouse_y):
#             break
#         mouse_clicked = False

#     cv2.imshow("Deadlift Form Tracker", combined_with_graph)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # -------------------- Save Data and Generate Reports --------------------
# if rep_data:
#     # Save CSV
#     with open(csv_file, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(csv_fields)
#         writer.writerows(rep_data)

#     # Create comprehensive graphs from CSV data
#     df = pd.read_csv(csv_file)
    
#     # Generate analysis graphs
#     plt.figure(figsize=(15, 10))
    
#     # Plot 1: Form Quality Progression
#     plt.subplot(2, 3, 1)
#     plt.plot(df['rep'], df['back_angle_quality'], 'ro-', label='Back Angle Quality')
#     plt.plot(df['rep'], df['hip_knee_alignment'], 'bo-', label='Hip-Knee Alignment')
#     plt.plot(df['rep'], df['rom_quality'], 'go-', label='ROM Quality')
#     plt.xlabel('Rep Number')
#     plt.ylabel('Quality Score (%)')
#     plt.title('Form Quality Progression')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     # Plot 2: Velocity Analysis
#     plt.subplot(2, 3, 2)
#     plt.bar(df['rep'], df['peak_velocity'], alpha=0.7, label='Peak Velocity')
#     plt.bar(df['rep'], df['mean_velocity'], alpha=0.7, label='Mean Velocity')
#     plt.xlabel('Rep Number')
#     plt.ylabel('Velocity (px/frame)')
#     plt.title('Velocity Analysis by Rep')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     # Plot 3: Biomechanical Loads
#     plt.subplot(2, 3, 3)
#     x_pos = np.arange(len(df))
#     width = 0.25
#     plt.bar(x_pos - width, df['spinal_compression'], width, label='Spinal Load')
#     plt.bar(x_pos, df['hip_moment'], width, label='Hip Moment')
#     plt.bar(x_pos + width, df['knee_shear'], width, label='Knee Shear')
#     plt.xlabel('Rep Number')
#     plt.ylabel('Relative Load')
#     plt.title('Biomechanical Loads by Rep')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     # Plot 4: Duration and Speed
#     plt.subplot(2, 3, 4)
#     plt.plot(df['rep'], df['duration_sec'], 'purple', marker='o', label='Duration')
#     plt.xlabel('Rep Number')
#     plt.ylabel('Duration (seconds)')
#     plt.title('Rep Duration Over Time')
#     plt.grid(True, alpha=0.3)
    
#     # Plot 5: Joint Angles
#     plt.subplot(2, 3, 5)
#     plt.plot(df['rep'], df['avg_torso_angle'], 'r-', marker='o', label='Torso Angle')
#     plt.plot(df['rep'], df['avg_hip_angle'], 'g-', marker='o', label='Hip Angle')
#     plt.plot(df['rep'], df['avg_knee_angle'], 'b-', marker='o', label='Knee Angle')
#     plt.xlabel('Rep Number')
#     plt.ylabel('Angle (degrees)')
#     plt.title('Average Joint Angles by Rep')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     # Plot 6: Overall Performance Score
#     plt.subplot(2, 3, 6)
#     overall_score = (df['back_angle_quality'] + df['hip_knee_alignment'] + df['rom_quality']) / 3
#     plt.plot(df['rep'], overall_score, 'orange', marker='s', linewidth=2, markersize=8)
#     plt.xlabel('Rep Number')
#     plt.ylabel('Overall Score (%)')
#     plt.title('Overall Performance Score')
#     plt.ylim(0, 100)
#     plt.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig('deadlift_analysis_report.png', dpi=300, bbox_inches='tight')
#     plt.show()
    
#     print(f"\n✓ Rep data logged. CSV saved: {csv_file}")
#     print(f"✓ Comprehensive analysis report saved: deadlift_analysis_report.png")
# else:
#     print("\nℹ️ No reps logged. CSV file not created.")

# # Cleanup
# cap.release()
# cv2.destroyAllWindows()
# pose.close()
# segmentation.close()
# print("\n✓ Advanced deadlift analysis session complete!")


import cv2
import mediapipe as mp
import numpy as np
import time
import math
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from collections import deque
import pandas as pd
from datetime import datetime
import os

# -------------------- Initialize MediaPipe --------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# -------------------- Camera Setup --------------------
working_camera = None
for index in range(10):
    temp_cap = cv2.VideoCapture(index)
    time.sleep(0.3)
    if temp_cap.isOpened():
        ret, frame = temp_cap.read()
        if ret and frame is not None:
            working_camera = index
            temp_cap.release()
            break
    temp_cap.release()

if working_camera is None:
    print("❌ No working camera found.")
    exit(1)

cap = cv2.VideoCapture(working_camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# -------------------- Video Recording Setup --------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"deadlift_session_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

video_filename = os.path.join(output_dir, f"deadlift_recording_{timestamp}.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30.0
frame_size = (1280 * 2, 720)  # Combined frame size
video_writer = None

# -------------------- CSV Setup --------------------
csv_file = os.path.join(output_dir, f"deadlift_reps_log_{timestamp}.csv")
csv_fields = [
    "rep", "start_time", "end_time", "duration_sec", "side",
    "avg_torso_angle", "min_torso_angle", "max_torso_angle",
    "avg_hip_angle", "min_hip_angle", "max_hip_angle",
    "avg_knee_angle", "min_knee_angle", "max_knee_angle",
    "avg_hip_alignment", "min_hip_alignment", "hip_displacement",
    "rom_quality", "rep_speed", "back_angle_quality", "hip_knee_alignment",
    "peak_velocity", "mean_velocity", "sticking_point_velocity",
    "spinal_compression", "hip_moment", "knee_shear",
    "work_efficiency", "technical_consistency", "fatigue_index"
]
rep_data = []

# -------------------- Trajectory Data Storage --------------------
rep_trajectories = []  # Store trajectories for each completed rep

# -------------------- Advanced Biomechanical Analysis --------------------
class BiomechanicalAnalyzer:
    def __init__(self):
        self.body_weight = 75  # kg - default, can be user configured
        self.barbell_weight = 60  # kg - default
        
    def calculate_spinal_loading(self, landmarks, width, height):
        """Estimate spinal compression forces"""
        hip_px = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * width),
                  int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * height))
        shoulder_px = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width),
                       int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height))
        
        # Calculate back angle
        back_angle = math.atan2(shoulder_px[0]-hip_px[0], shoulder_px[1]-hip_px[1])
        back_angle_deg = math.degrees(back_angle)
        
        # Simplified spinal compression calculation
        compression_force = self.barbell_weight * (1 + abs(math.sin(back_angle)))
        return compression_force
    
    def calculate_hip_moment(self, landmarks, width, height):
        """Calculate torque around hip joint"""
        hip_px = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * width),
                  int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * height))
        shoulder_px = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width),
                       int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height))
        
        # Torso length estimation
        torso_length = math.hypot(shoulder_px[0]-hip_px[0], shoulder_px[1]-hip_px[1])
        
        # Hip moment increases with forward lean
        moment_arm = torso_length * abs(math.sin(math.radians(self.calculate_torso_angle(landmarks, width, height))))
        hip_moment = self.body_weight * moment_arm
        return hip_moment
    
    def calculate_knee_shear(self, landmarks, width, height):
        """Estimate anterior shear forces on knees"""
        knee_px = (int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * width),
                   int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * height))
        ankle_px = (int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * width),
                    int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * height))
        
        # Shank angle
        shank_angle = math.atan2(knee_px[0]-ankle_px[0], knee_px[1]-ankle_px[1])
        
        # Shear force increases with horizontal shank position
        shear_force = self.barbell_weight * abs(math.cos(shank_angle))
        return shear_force
    
    def calculate_torso_angle(self, landmarks, width, height):
        """Calculate torso angle with vertical"""
        hip_px = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * width),
                  int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * height))
        shoulder_px = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width),
                       int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height))
        
        return angle_with_vertical(shoulder_px[0]-hip_px[0], shoulder_px[1]-hip_px[1])

# -------------------- Velocity Analysis --------------------
class VelocityAnalyzer:
    def __init__(self):
        self.velocity_thresholds = {
            'explosive': 1.0,  # m/s
            'moderate': 0.5,   # m/s  
            'slow': 0.3,       # m/s
            'grinding': 0.1    # m/s
        }
        self.bar_path_history = deque(maxlen=100)
        self.timestamps = deque(maxlen=100)
    
    def analyze_velocity(self, bar_position, timestamp):
        """Analyze bar velocity throughout movement"""
        self.bar_path_history.append(bar_position)
        self.timestamps.append(timestamp)
        
        if len(self.bar_path_history) < 2:
            return None
            
        # Calculate instantaneous velocity
        current_pos = np.array(self.bar_path_history[-1])
        prev_pos = np.array(self.bar_path_history[-2])
        time_diff = self.timestamps[-1] - self.timestamps[-2]
        
        if time_diff <= 0:
            return None
            
        displacement = np.linalg.norm(current_pos - prev_pos)
        velocity = displacement / time_diff
        
        return velocity
    
    def get_velocity_profile(self):
        """Get complete velocity profile"""
        if len(self.bar_path_history) < 2:
            return None
            
        velocities = []
        for i in range(1, len(self.bar_path_history)):
            current_pos = np.array(self.bar_path_history[i])
            prev_pos = np.array(self.bar_path_history[i-1])
            time_diff = self.timestamps[i] - self.timestamps[i-1]
            
            if time_diff > 0:
                displacement = np.linalg.norm(current_pos - prev_pos)
                velocities.append(displacement / time_diff)
        
        if not velocities:
            return None
            
        return {
            'current_velocity': velocities[-1] if velocities else 0,
            'peak_velocity': max(velocities) if velocities else 0,
            'mean_velocity': np.mean(velocities) if velocities else 0,
            'velocity_profile': velocities
        }

# -------------------- Fault Detection --------------------
class FaultDetector:
    def __init__(self):
        self.fault_thresholds = {
            'spinal_flexion': 15,  # degrees from neutral
            'hip_rise_lead': 0.2,  # seconds
            'knee_valgus': 10,     # degrees from vertical
            'asymmetry': 5,        # percentage difference
        }
    
    def detect_rounded_back(self, spine_angle, phase):
        """Detect thoracic and lumbar flexion"""
        faults = []
        
        if phase == 'descent' and spine_angle > self.fault_thresholds['spinal_flexion']:
            faults.append({
                'type': 'LUMBAR_FLEXION',
                'severity': spine_angle / self.fault_thresholds['spinal_flexion'],
                'phase': phase,
                'correction': 'Brace core and maintain neutral spine'
            })
        
        return faults
    
    def detect_knee_valgus(self, knee_px, ankle_px):
        """Detect knee cave (valgus collapse)"""
        # Calculate angle from vertical
        angle = abs(math.degrees(math.atan2(knee_px[0] - ankle_px[0], knee_px[1] - ankle_px[1])))
        
        if angle > self.fault_thresholds['knee_valgus']:
            return {
                'type': 'KNEE_VALGUS',
                'severity': angle / self.fault_thresholds['knee_valgus'],
                'max_angle': angle,
                'correction': 'Drive knees out, engage glutes'
            }
        return None

# -------------------- Enhanced Helper Functions --------------------
def lm_to_pixel(landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)

def angle_with_vertical(vx, vy):
    # Vector for vertical (0, -1)
    dot = vx * 0 + vy * (-1)
    mag_v = math.hypot(vx, vy)
    if mag_v == 0:
        return 0.0
    # Normalize dot product and clamp to [-1, 1] for safe acos
    cos_a = max(min(dot / mag_v, 1), -1)
    return math.degrees(math.acos(cos_a))

def angle_between_points(a, b, c):
    ba = np.array([a[0]-b[0], a[1]-b[1]])
    bc = np.array([c[0]-b[0], c[1]-b[1]])
    # Add a small epsilon to prevent division by zero
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    cos_angle = max(min(cos_angle,1),-1)
    return math.degrees(np.arccos(cos_angle))

def calculate_deadlift_reference_points(ankle_px, knee_px, shoulder_px, hip_px, frame_height):
    """
    Calculate reference points for proper deadlift form using plus sign system
    """
    # Reference 1: Vertical line from mid-foot (critical for bar path)
    mid_foot_x = ankle_px[0]
    
    # Reference 2: Horizontal line for optimal hip height
    knee_height = knee_px[1]
    shoulder_height = shoulder_px[1]
    optimal_hip_height = knee_height + (shoulder_height - knee_height) * 0.6
    
    # Reference 3: Back angle reference line
    back_angle_ref_y = ankle_px[1] - frame_height * 0.3
    
    return {
        'vertical_line_x': mid_foot_x,
        'optimal_hip_height': optimal_hip_height,
        'back_angle_ref': back_angle_ref_y,
        'mid_foot': (mid_foot_x, ankle_px[1])
    }

def draw_plus_sign_reference(img, reference_points, size=150, color=(0, 255, 255), thickness=3):
    """
    Draw enhanced plus sign reference system for deadlift form
    """
    vx = reference_points['vertical_line_x']
    hip_y = int(reference_points['optimal_hip_height'])
    mid_foot = reference_points['mid_foot']
    
    # Vertical reference line (bar path should follow this)
    cv2.line(img, (vx, 0), (vx, img.shape[0]), color, thickness-1)
    
    # Horizontal reference line (optimal hip height)
    cv2.line(img, (vx-size, hip_y), (vx+size, hip_y), color, thickness-1)
    
    # Plus sign center
    center_x, center_y = vx, hip_y
    
    # Enhanced plus sign with crosshair
    arm_length = size // 2
    # Vertical arm
    cv2.line(img, (center_x, center_y - arm_length), (center_x, center_y + arm_length), 
             (255, 255, 255), thickness+1)
    # Horizontal arm  
    cv2.line(img, (center_x - arm_length, center_y), (center_x + arm_length, center_y), 
             (255, 255, 255), thickness+1)
    
    # Outer circle
    cv2.circle(img, (center_x, center_y), arm_length, color, thickness)
    
    # Inner targeting circle
    cv2.circle(img, (center_x, center_y), 15, (0, 0, 255), -1)
    
    # Labels
    cv2.putText(img, "IDEAL HIP HEIGHT", (center_x + 40, center_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(img, "BAR PATH", (center_x + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Draw mid-foot marker
    cv2.circle(img, mid_foot, 8, (0, 255, 0), -1)
    cv2.putText(img, "MID-FOOT", (mid_foot[0] - 40, mid_foot[1] + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

def calculate_form_quality(hip_px, knee_px, shoulder_px, ankle_px, reference_points):
    """
    Calculate comprehensive form quality metrics using reference system
    """
    vx = reference_points['vertical_line_x']
    optimal_hip_y = reference_points['optimal_hip_height']
    
    # 1. Hip alignment quality (vertical reference)
    hip_v_alignment = 100 - min(100, (abs(hip_px[0] - vx) / (vx * 0.2 + 1e-6)) * 100)
    
    # 2. Hip height quality (horizontal reference)
    hip_h_alignment = 100 - min(100, (abs(hip_px[1] - optimal_hip_y) / (optimal_hip_y * 0.3 + 1e-6)) * 100)
    
    # 3. Back angle quality (should be ~45-60 degrees at setup)
    back_angle = angle_with_vertical(shoulder_px[0]-hip_px[0], shoulder_px[1]-hip_px[1])
    back_quality = 100 - min(100, abs(back_angle - 50) / 50 * 100)
    
    # 4. Hip-knee-ankle alignment (proportions)
    hip_knee_dist = math.hypot(hip_px[0]-knee_px[0], hip_px[1]-knee_px[1])
    knee_ankle_dist = math.hypot(knee_px[0]-ankle_px[0], knee_px[1]-ankle_px[1])
    alignment_ratio = hip_knee_dist / (knee_ankle_dist + 1e-6)
    alignment_quality = 100 - min(100, abs(alignment_ratio - 1.2) / 1.2 * 100)
    
    overall_quality = (hip_v_alignment + hip_h_alignment + back_quality + alignment_quality) / 4
    
    return {
        'overall': overall_quality,
        'hip_vertical': hip_v_alignment,
        'hip_height': hip_h_alignment,
        'back_angle': back_quality,
        'hip_knee_alignment': alignment_quality
    }

def draw_form_feedback(img, hip_px, form_quality, reference_points):
    """
    Draw real-time form feedback based on reference system
    """
    vx = reference_points['vertical_line_x']
    optimal_y = int(reference_points['optimal_hip_height'])
    
    # Draw current hip position relative to reference
    cv2.circle(img, hip_px, 12, (0, 0, 255), -1)
    cv2.circle(img, hip_px, 12, (255, 255, 255), 2)
    
    # Draw connection lines to reference
    cv2.line(img, hip_px, (vx, hip_px[1]), (255, 100, 100), 2)  # Horizontal connection
    cv2.line(img, hip_px, (hip_px[0], optimal_y), (255, 100, 100), 2)  # Vertical connection
    
    # Quality indicator color coding
    if form_quality['overall'] > 80:
        quality_color = (0, 255, 0)
    elif form_quality['overall'] > 60:
        quality_color = (0, 255, 255)
    else:
        quality_color = (0, 0, 255)
    
    # Draw quality bar
    bar_x, bar_y = 20, img.shape[0] - 100
    bar_width = 200
    bar_height = 20
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + int(bar_width * form_quality['overall'] / 100), bar_y + bar_height), quality_color, -1)
    cv2.putText(img, f"FORM: {form_quality['overall']:.0f}%", (bar_x, bar_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, quality_color, 2)

def draw_trajectory_analysis(img, trajectories, current_landmarks, width, height):
    """
    Draw real-time trajectory analysis with lines and paths
    """
    # Define colors for different trajectories
    colors = {
        'hip': (0, 255, 255),    # Cyan
        'shoulder': (255, 255, 0), # Yellow
        'knee': (255, 0, 255),   # Magenta
        'ankle': (0, 255, 0)     # Green
    }
    
    # Draw trajectories
    for landmark_name, points in trajectories.items():
        if len(points) > 1:
            color = colors.get(landmark_name, (255, 255, 255))
            # Draw trajectory line
            for i in range(1, len(points)):
                cv2.line(img, points[i-1], points[i], color, 2)
            # Draw current position
            if points:
                cv2.circle(img, points[-1], 8, color, -1)
    
    # Draw current landmark connections
    if current_landmarks:
        # Hip to knee
        cv2.line(img, current_landmarks['hip'], current_landmarks['knee'], (0, 200, 255), 3)
        # Knee to ankle
        cv2.line(img, current_landmarks['knee'], current_landmarks['ankle'], (0, 200, 255), 3)
        # Hip to shoulder
        cv2.line(img, current_landmarks['hip'], current_landmarks['shoulder'], (255, 200, 0), 3)
        
        # Label landmarks
        for landmark_name, point in current_landmarks.items():
            color = colors.get(landmark_name, (255, 255, 255))
            cv2.putText(img, landmark_name.upper(), (point[0] + 10, point[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def create_live_graphs(angle_history, velocity_history, biomech_history):
    """
    Create live matplotlib graphs for analysis
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Deadlift Analysis - Live Metrics', fontsize=14, fontweight='bold')
    
    # Plot 1: Joint Angles
    if angle_history:
        frames = list(range(len(angle_history)))
        ax1.clear()
        ax1.plot(frames, [a['torso'] for a in angle_history], 'r-', label='Torso', linewidth=2)
        ax1.plot(frames, [a['hip'] for a in angle_history], 'g-', label='Hip', linewidth=2)
        ax1.plot(frames, [a['knee'] for a in angle_history], 'b-', label='Knee', linewidth=2)
        ax1.set_title('Joint Angles Over Time')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Angle (degrees)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Velocity Profile
    if velocity_history:
        frames = list(range(len(velocity_history)))
        ax2.clear()
        ax2.plot(frames, velocity_history, 'purple', linewidth=2)
        ax2.set_title('Bar Velocity Profile')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Velocity (pixels/frame)')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Biomechanical Loads
    if biomech_history:
        frames = list(range(len(biomech_history)))
        ax3.clear()
        ax3.plot(frames, [b['spinal'] for b in biomech_history], 'red', label='Spinal Load', linewidth=2)
        ax3.plot(frames, [b['hip'] for b in biomech_history], 'blue', label='Hip Moment', linewidth=2)
        ax3.plot(frames, [b['knee'] for b in biomech_history], 'green', label='Knee Shear', linewidth=2)
        ax3.set_title('Biomechanical Loads')
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Relative Load')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Form Quality
    if angle_history and 'form_quality' in angle_history[0]:
        frames = list(range(len(angle_history)))
        ax4.clear()
        ax4.plot(frames, [a['form_quality'] for a in angle_history], 'orange', linewidth=2)
        ax4.set_title('Form Quality Over Time')
        ax4.set_xlabel('Frame')
        ax4.set_ylabel('Quality Score (%)')
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Convert matplotlib figure to OpenCV image
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    graph_img = np.asarray(buf)
    graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGBA2BGR)
    
    plt.close(fig)
    return graph_img

# -------------------- Classes and Rep Detection --------------------
class Button:
    def __init__(self, x, y, w, h, text, color, text_color):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.text = text
        self.color = color
        self.text_color = text_color
        self.hovered = False
    
    def draw(self, img):
        # Slightly brighten color on hover
        color = tuple(int(c * 1.2) if c * 1.2 <= 255 else 255 for c in self.color) if self.hovered else self.color
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), color, -1)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (255, 255, 255), 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        text_size = cv2.getTextSize(self.text, font, font_scale, thickness)[0]
        text_x = self.x + (self.w - text_size[0]) // 2
        text_y = self.y + (self.h + text_size[1]) // 2
        cv2.putText(img, self.text, (text_x, text_y), font, font_scale, self.text_color, thickness)
    
    def is_clicked(self, x, y):
        return self.x <= x <= self.x + self.w and self.y <= y <= self.y + self.h

class DeadliftRepDetector:
    def __init__(self):
        self.state = "STANDING"
        self.rep_count = 0
        self.hip_trajectory = deque(maxlen=100)
        self.current_rep_metrics = []
        self.current_rep_trajectory = {'hip': [], 'shoulder': [], 'knee': [], 'ankle': []}
        self.rep_start_time = None
        # Angle thresholds are for a side-view deadlift where 180 is straight up (STANDING)
        self.standing_threshold = 160
        self.bottom_threshold = 80
        # Normalised Y-coordinate change (0 to 1)
        self.hip_height_threshold = 0.005 # Reduced sensitivity
        self.last_hip_y = None
        
    def update(self, hip_y_norm, torso_angle, hip_angle, knee_angle, form_quality):
        rep_completed = False
        
        if self.last_hip_y is None:
            self.last_hip_y = hip_y_norm
            return rep_completed, self.state
        
        # Hip Y-coordinate increases as the hip moves down (MediaPipe convention)
        hip_moving_down = hip_y_norm > self.last_hip_y + self.hip_height_threshold
        hip_moving_up = hip_y_norm < self.last_hip_y - self.hip_height_threshold
        
        if self.state == "STANDING":
            # Start of descent: Torso bends AND hip is moving down
            if torso_angle < self.standing_threshold - 10 and hip_moving_down:
                self.state = "DESCENDING"
                self.rep_start_time = time.time()
                self.current_rep_metrics = []
                self.current_rep_trajectory = {'hip': [], 'shoulder': [], 'knee': [], 'ankle': []}
                print(f"  → Rep {self.rep_count + 1} started (Descending)")
                
        elif self.state == "DESCENDING":
            # Reached bottom: Torso is bent low AND hip angle is small
            if torso_angle < self.bottom_threshold and hip_angle < 110:
                self.state = "BOTTOM"
                print(f"  → Bottom position reached")
                
        elif self.state == "BOTTOM":
            # Start of ascent: Hip is moving up AND torso is extending
            if hip_moving_up and torso_angle > self.bottom_threshold + 5:
                self.state = "ASCENDING"
                print(f"  → Ascending")
                
        elif self.state == "ASCENDING":
            # Rep complete: Torso is upright AND hip is fully extended
            if torso_angle > self.standing_threshold - 5 and hip_angle > 165:
                self.state = "STANDING"
                self.rep_count += 1
                rep_completed = True
                print(f"✓ Rep {self.rep_count} COMPLETED!")
        
        self.last_hip_y = hip_y_norm
        return rep_completed, self.state
    
    def add_metrics(self, torso, hip, knee, form_metrics, biomech_metrics, velocity_metrics):
        self.current_rep_metrics.append([torso, hip, knee, form_metrics, biomech_metrics, velocity_metrics])
    
    def add_trajectory_point(self, landmarks_dict):
        """Store trajectory points for current rep"""
        for key, point in landmarks_dict.items():
            self.current_rep_trajectory[key].append(point)
    
    def get_rep_summary(self):
        if not self.current_rep_metrics:
            return None
        
        metrics = np.array([m[:3] for m in self.current_rep_metrics])
        form_metrics = [m[3] for m in self.current_rep_metrics]
        biomech_metrics = [m[4] for m in self.current_rep_metrics]
        velocity_metrics = [m[5] for m in self.current_rep_metrics]
        
        duration = time.time() - self.rep_start_time if self.rep_start_time else 0
        
        avg_form_quality = np.mean([fm['overall'] for fm in form_metrics])
        avg_back_quality = np.mean([fm['back_angle'] for fm in form_metrics])
        avg_alignment_quality = np.mean([fm['hip_knee_alignment'] for fm in form_metrics])
        
        # Biomechanical averages
        avg_spinal_load = np.mean([bm['spinal'] for bm in biomech_metrics])
        avg_hip_moment = np.mean([bm['hip'] for bm in biomech_metrics])
        avg_knee_shear = np.mean([bm['knee'] for bm in biomech_metrics])
        
        # Velocity analysis
        peak_velocity = max([vm.get('current_velocity', 0) for vm in velocity_metrics if vm]) if any(velocity_metrics) else 0
        mean_velocity = np.mean([vm.get('current_velocity', 0) for vm in velocity_metrics if vm]) if any(velocity_metrics) else 0
        
        return {
            'torso': (np.mean(metrics[:, 0]), np.min(metrics[:, 0]), np.max(metrics[:, 0])),
            'hip': (np.mean(metrics[:, 1]), np.min(metrics[:, 1]), np.max(metrics[:, 1])),
            'knee': (np.mean(metrics[:, 2]), np.min(metrics[:, 2]), np.max(metrics[:, 2])),
            'form_quality': avg_form_quality,
            'back_quality': avg_back_quality,
            'alignment_quality': avg_alignment_quality,
            'duration': duration,
            'rom_quality': np.max(metrics[:, 0]) - np.min(metrics[:, 0]),
            'spinal_load': avg_spinal_load,
            'hip_moment': avg_hip_moment,
            'knee_shear': avg_knee_shear,
            'peak_velocity': peak_velocity,
            'mean_velocity': mean_velocity
        }
    
    def get_current_trajectory(self):
        """Return the trajectory data for the current completed rep"""
        return self.current_rep_trajectory.copy()

# -------------------- Initialize Systems --------------------
rep_detector = DeadliftRepDetector()
biomech_analyzer = BiomechanicalAnalyzer()
velocity_analyzer = VelocityAnalyzer()
fault_detector = FaultDetector()

# Trajectory tracking
trajectories = {
    'hip': deque(maxlen=50),
    'shoulder': deque(maxlen=50),
    'knee': deque(maxlen=50),
    'ankle': deque(maxlen=50)
}

# History for graphs
angle_history = deque(maxlen=100)
velocity_history = deque(maxlen=100)
biomech_history = deque(maxlen=100)

hip_trajectory = deque(maxlen=150)
blur_enabled = True
logging_active = False
show_reference = True
show_trajectory = True
show_graphs = True
recording_active = False

# Create buttons
button_y = 10
button_spacing = 10
button_width = 100
button_height = 40

start_btn = Button(10, button_y, button_width, button_height, "START", (0, 200, 0), (255, 255, 255))
stop_btn = Button(10 + button_width + button_spacing, button_y, button_width, button_height, "STOP", (0, 0, 200), (255, 255, 255))
blur_btn = Button(10 + 2*(button_width + button_spacing), button_y, button_width, button_height, "BLUR ON", (200, 100, 0), (255, 255, 255))
ref_btn = Button(10 + 3*(button_width + button_spacing), button_y, button_width, button_height, "REF ON", (100, 100, 200), (255, 255, 255))
traj_btn = Button(10 + 4*(button_width + button_spacing), button_y, button_width, button_height, "TRAJ ON", (150, 150, 0), (255, 255, 255))
graph_btn = Button(10 + 5*(button_width + button_spacing), button_y, button_width, button_height, "GRAPHS ON", (200, 100, 200), (255, 255, 255))
quit_btn = Button(10 + 6*(button_width + button_spacing), button_y, button_width, button_height, "QUIT", (200, 0, 0), (255, 255, 255))

buttons = [start_btn, stop_btn, blur_btn, ref_btn, traj_btn, graph_btn, quit_btn]

# -------------------- Mouse Callback --------------------
mouse_x, mouse_y = 0, 0
mouse_clicked = False

def mouse_callback(event, x, y, flags, param):
    """Handles mouse events for button clicks and hover."""
    global mouse_x, mouse_y, mouse_clicked
    mouse_x, mouse_y = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_clicked = True

# -------------------- Window Initialization --------------------
cv2.namedWindow("Deadlift Form Tracker")
cv2.setMouseCallback("Deadlift Form Tracker", mouse_callback)

print("\n" + "="*60)
print("ADVANCED DEADLIFT FORM TRACKER - Professional Biomechanical Analysis")
print("="*60)
print("Features:")
print("  ⊕ Real-time trajectory analysis with colored paths")
print("  ⊕ Live biomechanical load monitoring")
print("  ⊕ Velocity-based performance analysis") 
print("  ⊕ Comprehensive graphing and CSV logging")
print("  ⊕ Video recording with rep-by-rep analysis")
print(f"  ⊕ Output directory: {output_dir}")
print("="*60)

# -------------------- Main Loop --------------------
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        if working_camera is not None:
            print("⚠️ Warning: Camera frame retrieval failed. Exiting...")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if blur_enabled:
        # Selfie segmentation for background blur
        seg_results = segmentation.process(rgb_frame)
        mask = seg_results.segmentation_mask
        condition = np.stack((mask,)*3, axis=-1) > 0.5
        blurred_frame = cv2.GaussianBlur(frame, (55,55), 0)
        frame = np.where(condition, frame, blurred_frame)

    # Black canvas for skeleton visualization
    black_canvas = np.zeros((h, w, 3), dtype=np.uint8)

    pose_results = pose.process(rgb_frame)
    
    current_landmarks = None
    current_timestamp = time.time()
    
    if pose_results.pose_landmarks:
        lm = pose_results.pose_landmarks.landmark
        
        # Determine which side (left/right) is more visible for side-view tracking
        left_hip_vis = lm[mp_pose.PoseLandmark.LEFT_HIP].visibility
        right_hip_vis = lm[mp_pose.PoseLandmark.RIGHT_HIP].visibility
        use_left = left_hip_vis >= right_hip_vis and left_hip_vis > 0.5
        use_right = right_hip_vis > left_hip_vis and right_hip_vis > 0.5
        
        if use_left or use_right:
            side_name = "LEFT" if use_left else "RIGHT"
            
            if use_left:
                hip_idx, knee_idx, ankle_idx, shoulder_idx = (
                    mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE,
                    mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_SHOULDER
                )
            else:
                hip_idx, knee_idx, ankle_idx, shoulder_idx = (
                    mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE,
                    mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_SHOULDER
                )

            # Get pixel coordinates
            hip_px = lm_to_pixel(lm[hip_idx], w, h)
            knee_px = lm_to_pixel(lm[knee_idx], w, h)
            ankle_px = lm_to_pixel(lm[ankle_idx], w, h)
            shoulder_px = lm_to_pixel(lm[shoulder_idx], w, h)

            # Update trajectories
            trajectories['hip'].append(hip_px)
            trajectories['shoulder'].append(shoulder_px)
            trajectories['knee'].append(knee_px)
            trajectories['ankle'].append(ankle_px)
            
            current_landmarks = {
                'hip': hip_px,
                'shoulder': shoulder_px,
                'knee': knee_px,
                'ankle': ankle_px
            }

            # Calculate enhanced reference points
            reference_points = calculate_deadlift_reference_points(
                ankle_px, knee_px, shoulder_px, hip_px, h
            )
            
            # Calculate comprehensive form quality
            form_quality = calculate_form_quality(
                hip_px, knee_px, shoulder_px, ankle_px, reference_points
            )

            # Calculate angles
            torso_angle = angle_with_vertical(shoulder_px[0]-hip_px[0], shoulder_px[1]-hip_px[1])
            hip_angle = angle_between_points(shoulder_px, hip_px, knee_px)
            knee_angle = angle_between_points(hip_px, knee_px, ankle_px)

            # Biomechanical analysis
            spinal_load = biomech_analyzer.calculate_spinal_loading(lm, w, h)
            hip_moment = biomech_analyzer.calculate_hip_moment(lm, w, h)
            knee_shear = biomech_analyzer.calculate_knee_shear(lm, w, h)
            
            biomech_metrics = {
                'spinal': spinal_load,
                'hip': hip_moment,
                'knee': knee_shear
            }

            # Velocity analysis (using hip as proxy for bar path)
            velocity_metrics = velocity_analyzer.analyze_velocity(hip_px, current_timestamp)
            velocity_profile = velocity_analyzer.get_velocity_profile()
            
            # Update history for graphs
            angle_history.append({
                'torso': torso_angle,
                'hip': hip_angle,
                'knee': knee_angle,
                'form_quality': form_quality['overall']
            })
            
            if velocity_profile:
                velocity_history.append(velocity_profile['current_velocity'])
            else:
                velocity_history.append(0)
                
            biomech_history.append(biomech_metrics)

            # Track hip trajectory
            hip_trajectory.append(hip_px)

            # Rep detection and logging
            if logging_active:
                # Add current frame's metrics
                rep_detector.add_metrics(torso_angle, hip_angle, knee_angle, form_quality, biomech_metrics, velocity_profile)
                rep_detector.add_trajectory_point(current_landmarks)
                
                # Check for rep state change
                rep_completed, current_state = rep_detector.update(
                    lm[hip_idx].y, torso_angle, hip_angle, knee_angle, form_quality
                )
                
                if rep_completed:
                    summary = rep_detector.get_rep_summary()
                    trajectory_data = rep_detector.get_current_trajectory()
                    
                    if summary:
                        # Store trajectory for this rep
                        rep_trajectories.append({
                            'rep_number': rep_detector.rep_count,
                            'trajectories': trajectory_data
                        })
                        
                        # Append the full set of logged data for the completed rep
                        rep_data.append([
                            rep_detector.rep_count,
                            rep_detector.rep_start_time,
                            time.time(),
                            summary['duration'],
                            side_name,
                            summary['torso'][0], summary['torso'][1], summary['torso'][2],
                            summary['hip'][0], summary['hip'][1], summary['hip'][2],
                            summary['knee'][0], summary['knee'][1], summary['knee'][2],
                            form_quality['hip_vertical'], form_quality['hip_height'],
                            abs(hip_px[0] - reference_points['vertical_line_x']),
                            summary['rom_quality'],
                            1.0 / summary['duration'] if summary['duration'] > 0 else 0,
                            summary['back_quality'],
                            summary['alignment_quality'],
                            summary['peak_velocity'],
                            summary['mean_velocity'],
                            0,  # sticking_point_velocity placeholder
                            summary['spinal_load'],
                            summary['hip_moment'], 
                            summary['knee_shear'],
                            85.0,  # work_efficiency placeholder
                            90.0,  # technical_consistency placeholder
                            0.1    # fatigue_index placeholder
                        ])

            # Draw skeleton
            for canvas in [frame, black_canvas]:
                # Draw connections and landmarks
                mp_drawing.draw_landmarks(
                    canvas, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(40,40,40), thickness=8, circle_radius=6),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(60,60,60), thickness=12)
                )
                mp_drawing.draw_landmarks(
                    canvas, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,200,255), thickness=3)
                )

            # Draw enhanced reference system
            if show_reference:
                for canvas in [frame, black_canvas]:
                    draw_plus_sign_reference(canvas, reference_points, size=200)
                    draw_form_feedback(canvas, hip_px, form_quality, reference_points)

            # Draw trajectory analysis
            if show_trajectory:
                for canvas in [frame, black_canvas]:
                    draw_trajectory_analysis(canvas, trajectories, current_landmarks, w, h)

            # Enhanced display info
            status_color = (0, 255, 0) if logging_active else (100, 100, 100)
            cv2.putText(frame, f"Reps: {rep_detector.rep_count} | State: {rep_detector.state}", 
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame, f"Form Quality: {form_quality['overall']:.0f}% | Back: {form_quality['back_angle']:.0f}%", 
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Hip Alignment: V{form_quality['hip_vertical']:.0f}% H{form_quality['hip_height']:.0f}%", 
                        (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
            
            # Biomechanical data display
            cv2.putText(frame, f"Spinal Load: {spinal_load:.1f}N | Hip Moment: {hip_moment:.1f}Nm", 
                        (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
            
            if velocity_profile:
                cv2.putText(frame, f"Velocity: {velocity_profile['current_velocity']:.2f} px/frame | Peak: {velocity_profile['peak_velocity']:.2f}", 
                            (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 255), 1)
        else:
            cv2.putText(frame, "ADJUST VIEW: Side-view body not clearly visible.", 
                        (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # Create and display graphs
    if show_graphs and (angle_history or velocity_history or biomech_history):
        graph_img = create_live_graphs(list(angle_history), list(velocity_history), list(biomech_history))
        graph_h, graph_w = graph_img.shape[:2]
        
        # Resize graph to fit alongside video
        graph_scale = h / graph_h * 0.8
        new_graph_w = int(graph_w * graph_scale)
        new_graph_h = int(graph_h * graph_scale)
        graph_img = cv2.resize(graph_img, (new_graph_w, new_graph_h))
        
        # Place graph to the right of the combined video
        combined = np.hstack((frame, black_canvas))
        if combined.shape[0] > new_graph_h:
            # Pad graph to match height
            padding = np.zeros((combined.shape[0] - new_graph_h, new_graph_w, 3), dtype=np.uint8)
            graph_img = np.vstack((graph_img, padding))
        
        combined_with_graph = np.hstack((combined, graph_img))
    else:
        combined_with_graph = np.hstack((frame, black_canvas))

    # Draw buttons and handle clicks
    for btn in buttons:
        btn.hovered = btn.is_clicked(mouse_x, mouse_y)
        btn.draw(combined_with_graph)

    # Update dynamic button text
    blur_btn.text = "BLUR ON" if blur_enabled else "BLUR OFF"
    ref_btn.text = "REF ON" if show_reference else "REF OFF"
    traj_btn.text = "TRAJ ON" if show_trajectory else "TRAJ OFF"
    graph_btn.text = "GRAPHS ON" if show_graphs else "GRAPHS OFF"

    # Recording indicator
    if recording_active:
        cv2.circle(combined_with_graph, (combined_with_graph.shape[1] - 30, 30), 10, (0, 0, 255), -1)
        cv2.putText(combined_with_graph, "REC", (combined_with_graph.shape[1] - 70, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if mouse_clicked:
        if start_btn.is_clicked(mouse_x, mouse_y):
            if not logging_active:
                logging_active = True
                recording_active = True
                
                # Initialize video writer
                if video_writer is None:
                    combined_shape = np.hstack((frame, black_canvas)).shape
                    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, 
                                                   (combined_shape[1], combined_shape[0]))
                
                print("✓ Tracking and recording started...")
        elif stop_btn.is_clicked(mouse_x, mouse_y):
            if logging_active:
                logging_active = False
                recording_active = False
                
                # Release video writer
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                    print(f"✓ Video saved: {video_filename}")
                
                print("✓ Tracking stopped...")
        elif blur_btn.is_clicked(mouse_x, mouse_y):
            blur_enabled = not blur_enabled
        elif ref_btn.is_clicked(mouse_x, mouse_y):
            show_reference = not show_reference
        elif traj_btn.is_clicked(mouse_x, mouse_y):
            show_trajectory = not show_trajectory
        elif graph_btn.is_clicked(mouse_x, mouse_y):
            show_graphs = not show_graphs
        elif quit_btn.is_clicked(mouse_x, mouse_y):
            break
        mouse_clicked = False

    # Write frame to video if recording
    if recording_active and video_writer is not None:
        combined_frame = np.hstack((frame, black_canvas))
        video_writer.write(combined_frame)

    cv2.imshow("Deadlift Form Tracker", combined_with_graph)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------- Save Data and Generate Reports --------------------
print("\n" + "="*60)
print("GENERATING COMPREHENSIVE ANALYSIS REPORTS...")
print("="*60)

if rep_data:
    # Save CSV
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_fields)
        writer.writerows(rep_data)
    print(f"✓ CSV data saved: {csv_file}")

    # Create comprehensive graphs from CSV data
    df = pd.read_csv(csv_file)
    
    # ==================== MAIN ANALYSIS DASHBOARD ====================
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Form Quality Progression
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(df['rep'], df['back_angle_quality'], 'ro-', label='Back Angle Quality', linewidth=2, markersize=8)
    ax1.plot(df['rep'], df['hip_knee_alignment'], 'bo-', label='Hip-Knee Alignment', linewidth=2, markersize=8)
    ax1.plot(df['rep'], df['rom_quality'], 'go-', label='ROM Quality', linewidth=2, markersize=8)
    ax1.set_xlabel('Rep Number', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Quality Score (%)', fontsize=10, fontweight='bold')
    ax1.set_title('Form Quality Progression', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Velocity Analysis
    ax2 = plt.subplot(3, 3, 2)
    x_pos = np.arange(len(df))
    width = 0.35
    ax2.bar(x_pos - width/2, df['peak_velocity'], width, alpha=0.8, label='Peak Velocity', color='#FF6B6B')
    ax2.bar(x_pos + width/2, df['mean_velocity'], width, alpha=0.8, label='Mean Velocity', color='#4ECDC4')
    ax2.set_xlabel('Rep Number', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Velocity (px/frame)', fontsize=10, fontweight='bold')
    ax2.set_title('Velocity Analysis by Rep', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(df['rep'])
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Biomechanical Loads
    ax3 = plt.subplot(3, 3, 3)
    x_pos = np.arange(len(df))
    width = 0.25
    ax3.bar(x_pos - width, df['spinal_compression'], width, label='Spinal Load', color='#FF6B6B', alpha=0.9)
    ax3.bar(x_pos, df['hip_moment'], width, label='Hip Moment', color='#95E1D3', alpha=0.9)
    ax3.bar(x_pos + width, df['knee_shear'], width, label='Knee Shear', color='#F38181', alpha=0.9)
    ax3.set_xlabel('Rep Number', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Relative Load', fontsize=10, fontweight='bold')
    ax3.set_title('Biomechanical Loads by Rep', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(df['rep'])
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Duration and Speed
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(df['rep'], df['duration_sec'], 'o-', color='#A8E6CF', linewidth=2.5, markersize=10, 
             markeredgecolor='white', markeredgewidth=2)
    ax4.set_xlabel('Rep Number', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Duration (seconds)', fontsize=10, fontweight='bold')
    ax4.set_title('Rep Duration Over Time', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.fill_between(df['rep'], df['duration_sec'], alpha=0.3, color='#A8E6CF')
    
    # Plot 5: Joint Angles
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(df['rep'], df['avg_torso_angle'], 'r-', marker='o', label='Torso Angle', linewidth=2.5, markersize=8)
    ax5.plot(df['rep'], df['avg_hip_angle'], 'g-', marker='s', label='Hip Angle', linewidth=2.5, markersize=8)
    ax5.plot(df['rep'], df['avg_knee_angle'], 'b-', marker='^', label='Knee Angle', linewidth=2.5, markersize=8)
    ax5.set_xlabel('Rep Number', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Angle (degrees)', fontsize=10, fontweight='bold')
    ax5.set_title('Average Joint Angles by Rep', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Overall Performance Score
    ax6 = plt.subplot(3, 3, 6)
    overall_score = (df['back_angle_quality'] + df['hip_knee_alignment'] + df['rom_quality']) / 3
    colors = ['#FF6B6B' if score < 70 else '#FFD93D' if score < 85 else '#6BCF7F' for score in overall_score]
    ax6.bar(df['rep'], overall_score, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    ax6.set_xlabel('Rep Number', fontsize=10, fontweight='bold')
    ax6.set_ylabel('Overall Score (%)', fontsize=10, fontweight='bold')
    ax6.set_title('Overall Performance Score', fontsize=12, fontweight='bold')
    ax6.set_ylim(0, 100)
    ax6.axhline(y=85, color='green', linestyle='--', alpha=0.5, label='Excellent (85%)')
    ax6.axhline(y=70, color='yellow', linestyle='--', alpha=0.5, label='Good (70%)')
    ax6.legend(fontsize=7)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Plot 7: Range of Motion Analysis
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(df['rep'], df['min_torso_angle'], 'r--', label='Min Torso', linewidth=2, alpha=0.7)
    ax7.plot(df['rep'], df['max_torso_angle'], 'r-', label='Max Torso', linewidth=2, alpha=0.7)
    ax7.fill_between(df['rep'], df['min_torso_angle'], df['max_torso_angle'], alpha=0.2, color='red')
    ax7.set_xlabel('Rep Number', fontsize=10, fontweight='bold')
    ax7.set_ylabel('Torso Angle Range (degrees)', fontsize=10, fontweight='bold')
    ax7.set_title('Range of Motion Analysis', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Hip Alignment Quality
    ax8 = plt.subplot(3, 3, 8)
    ax8.scatter(df['rep'], df['avg_hip_alignment'], s=200, c=df['avg_hip_alignment'], 
               cmap='RdYlGn', vmin=70, vmax=100, edgecolors='white', linewidths=2)
    ax8.set_xlabel('Rep Number', fontsize=10, fontweight='bold')
    ax8.set_ylabel('Hip Alignment Score', fontsize=10, fontweight='bold')
    ax8.set_title('Hip Alignment Quality', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax8.collections[0], ax=ax8)
    cbar.set_label('Quality Score', fontsize=8)
    
    # Plot 9: Rep Speed Consistency
    ax9 = plt.subplot(3, 3, 9)
    ax9.plot(df['rep'], df['rep_speed'], 'o-', color='#FF6B9D', linewidth=2.5, markersize=10,
             markeredgecolor='white', markeredgewidth=2)
    ax9.set_xlabel('Rep Number', fontsize=10, fontweight='bold')
    ax9.set_ylabel('Rep Speed (reps/sec)', fontsize=10, fontweight='bold')
    ax9.set_title('Rep Speed Consistency', fontsize=12, fontweight='bold')
    ax9.grid(True, alpha=0.3)
    mean_speed = df['rep_speed'].mean()
    ax9.axhline(y=mean_speed, color='cyan', linestyle='--', linewidth=2, label=f'Mean: {mean_speed:.2f}')
    ax9.legend(fontsize=8)
    
    plt.tight_layout()
    dashboard_filename = os.path.join(output_dir, f'deadlift_analysis_dashboard_{timestamp}.png')
    plt.savefig(dashboard_filename, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
    print(f"✓ Analysis dashboard saved: {dashboard_filename}")
    plt.close()
    
    # ==================== TRAJECTORY ANALYSIS PLOTS ====================
    if rep_trajectories:
        print("\nGenerating trajectory analysis plots...")
        
        # Normalize trajectories for better comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Joint Trajectory Analysis - All Reps', fontsize=16, fontweight='bold')
        
        # Define colormap for different reps
        colors = plt.cm.viridis(np.linspace(0, 1, len(rep_trajectories)))
        
        joint_names = ['hip', 'shoulder', 'knee', 'ankle']
        joint_labels = ['Hip Trajectory', 'Shoulder Trajectory', 'Knee Trajectory', 'Ankle Trajectory']
        
        for idx, (joint_name, joint_label) in enumerate(zip(joint_names, joint_labels)):
            ax = axes[idx // 2, idx % 2]
            
            for rep_idx, rep_data in enumerate(rep_trajectories):
                rep_num = rep_data['rep_number']
                trajectory = rep_data['trajectories'][joint_name]
                
                if len(trajectory) > 0:
                    # Normalize coordinates to range [0, 1]
                    x_coords = np.array([p[0] for p in trajectory]) / w
                    y_coords = np.array([p[1] for p in trajectory]) / h
                    
                    ax.plot(x_coords, y_coords, '-', color=colors[rep_idx], 
                           linewidth=2, alpha=0.8, label=f'Rep {rep_num}')
            
            ax.set_xlabel('X Position (normalized)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Y Position (normalized)', fontsize=10, fontweight='bold')
            ax.set_title(joint_label, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.invert_yaxis()  # Invert Y axis to match screen coordinates
            
            if len(rep_trajectories) <= 10:
                ax.legend(fontsize=8, loc='best')
        
        plt.tight_layout()
        trajectory_filename = os.path.join(output_dir, f'trajectory_analysis_{timestamp}.png')
        plt.savefig(trajectory_filename, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        print(f"✓ Trajectory analysis saved: {trajectory_filename}")
        plt.close()
        
        # ==================== INDIVIDUAL REP TRAJECTORY COMPARISON ====================
        if len(rep_trajectories) >= 2:
            fig, axes = plt.subplots(2, min(len(rep_trajectories), 4), figsize=(20, 10))
            fig.suptitle('Individual Rep Trajectory Comparison', fontsize=16, fontweight='bold')
            
            if len(rep_trajectories) < 4:
                axes = axes.reshape(2, -1)
            
            for rep_idx, rep_data in enumerate(rep_trajectories[:4]):  # Show first 4 reps
                rep_num = rep_data['rep_number']
                
                # Top row: Full body trajectory
                ax_top = axes[0, rep_idx] if len(rep_trajectories) >= 2 else axes[0]
                
                for joint_name, color in zip(['hip', 'shoulder', 'knee', 'ankle'], 
                                             ['cyan', 'yellow', 'magenta', 'lime']):
                    trajectory = rep_data['trajectories'][joint_name]
                    if len(trajectory) > 0:
                        x_coords = np.array([p[0] for p in trajectory]) / w
                        y_coords = np.array([p[1] for p in trajectory]) / h
                        ax_top.plot(x_coords, y_coords, '-', color=color, linewidth=2, 
                                   alpha=0.8, label=joint_name.capitalize())
                
                ax_top.set_title(f'Rep {rep_num} - Full Body', fontsize=11, fontweight='bold')
                ax_top.set_xlabel('X Position', fontsize=9)
                ax_top.set_ylabel('Y Position', fontsize=9)
                ax_top.legend(fontsize=7)
                ax_top.grid(True, alpha=0.3)
                ax_top.invert_yaxis()
                
                # Bottom row: Hip trajectory detail
                ax_bottom = axes[1, rep_idx] if len(rep_trajectories) >= 2 else axes[1]
                
                hip_trajectory = rep_data['trajectories']['hip']
                if len(hip_trajectory) > 0:
                    x_coords = np.array([p[0] for p in hip_trajectory]) / w
                    y_coords = np.array([p[1] for p in hip_trajectory]) / h
                    
                    # Create gradient color based on progression
                    points = np.array([x_coords, y_coords]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    
                    norm = plt.Normalize(0, len(x_coords))
                    lc = plt.matplotlib.collections.LineCollection(segments, cmap='viridis', 
                                                                   norm=norm, linewidth=3, alpha=0.9)
                    lc.set_array(np.arange(len(x_coords)))
                    ax_bottom.add_collection(lc)
                    
                    ax_bottom.scatter(x_coords[0], y_coords[0], s=200, c='green', 
                                     edgecolors='white', linewidths=2, label='Start', zorder=5)
                    ax_bottom.scatter(x_coords[-1], y_coords[-1], s=200, c='red', 
                                     edgecolors='white', linewidths=2, label='End', zorder=5)
                
                ax_bottom.set_title(f'Rep {rep_num} - Hip Path', fontsize=11, fontweight='bold')
                ax_bottom.set_xlabel('X Position', fontsize=9)
                ax_bottom.set_ylabel('Y Position', fontsize=9)
                ax_bottom.legend(fontsize=7)
                ax_bottom.grid(True, alpha=0.3)
                ax_bottom.invert_yaxis()
                ax_bottom.set_xlim([0, 1])
                ax_bottom.set_ylim([0, 1])
            
            plt.tight_layout()
            individual_traj_filename = os.path.join(output_dir, f'individual_rep_trajectories_{timestamp}.png')
            plt.savefig(individual_traj_filename, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
            print(f"✓ Individual rep trajectories saved: {individual_traj_filename}")
            plt.close()
    
    # ==================== SUMMARY STATISTICS ====================
    summary_stats = {
        'Total Reps': len(df),
        'Average Duration': f"{df['duration_sec'].mean():.2f}s",
        'Average Form Quality': f"{overall_score.mean():.1f}%",
        'Average Peak Velocity': f"{df['peak_velocity'].mean():.2f}",
        'Average Spinal Load': f"{df['spinal_compression'].mean():.1f}N",
        'Best Rep': int(df.loc[overall_score.idxmax(), 'rep']),
        'Best Form Score': f"{overall_score.max():.1f}%"
    }
    
    print("\n" + "="*60)
    print("SESSION SUMMARY STATISTICS")
    print("="*60)
    for key, value in summary_stats.items():
        print(f"{key:.<40} {value}")
    
    # Save summary to text file
    summary_filename = os.path.join(output_dir, f'session_summary_{timestamp}.txt')
    with open(summary_filename, 'w') as f:
        f.write("="*60 + "\n")
        f.write("DEADLIFT SESSION SUMMARY\n")
        f.write("="*60 + "\n\n")
        for key, value in summary_stats.items():
            f.write(f"{key}: {value}\n")
        f.write("\n" + "="*60 + "\n")
    
    print(f"✓ Summary statistics saved: {summary_filename}")
    
else:
    print("\nℹ️ No reps logged. Reports not generated.")

# Cleanup
if video_writer is not None:
    video_writer.release()

cap.release()
cv2.destroyAllWindows()
pose.close()
segmentation.close()

print("\n" + "="*60)
print("✓ Advanced deadlift analysis session complete!")
print(f"✓ All files saved to: {output_dir}")
print("="*60 + "\n")
