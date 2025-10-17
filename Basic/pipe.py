import cv2
import mediapipe as mp
import numpy as np
import time
import math
import csv
# Matplotlib is imported but not used in the provided code block for plotting
# from matplotlib.backends.backend_agg import FigureCanvasAgg
from collections import deque

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

# -------------------- CSV Setup --------------------
csv_file = "deadlift_reps_log.csv"
csv_fields = [
    "rep", "start_time", "end_time", "duration_sec", "side",
    "avg_torso_angle", "min_torso_angle", "max_torso_angle",
    "avg_hip_angle", "min_hip_angle", "max_hip_angle",
    "avg_knee_angle", "min_knee_angle", "max_knee_angle",
    "avg_hip_alignment", "min_hip_alignment", "hip_displacement",
    "rom_quality", "rep_speed", "back_angle_quality", "hip_knee_alignment"
]
rep_data = []

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
    Based on side view analysis for optimal deadlift positioning
    """
    # Reference 1: Vertical line from mid-foot (critical for bar path)
    mid_foot_x = ankle_px[0]
    
    # Reference 2: Horizontal line for optimal hip height
    # In proper deadlift, hips should be higher than knees but lower than shoulders
    knee_height = knee_px[1]
    shoulder_height = shoulder_px[1]
    # Optimal hip height is set at 60% of the distance between the knee and shoulder vertically
    optimal_hip_height = knee_height + (shoulder_height - knee_height) * 0.6
    
    # Reference 3: Back angle reference line (45-60 degrees at setup) - Not used for drawing here, but kept for context.
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
    hip_y = int(reference_points['optimal_hip_height']) # Ensure integer for drawing
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
    # Deviation from vertical line, normalized by 20% of the x-coordinate (arbitrary but relative tolerance)
    hip_v_alignment = 100 - min(100, (abs(hip_px[0] - vx) / (vx * 0.2 + 1e-6)) * 100)
    
    # 2. Hip height quality (horizontal reference)
    # Deviation from optimal hip height, normalized by 30% of optimal_hip_y (relative tolerance)
    hip_h_alignment = 100 - min(100, (abs(hip_px[1] - optimal_hip_y) / (optimal_hip_y * 0.3 + 1e-6)) * 100)
    
    # 3. Back angle quality (should be ~45-60 degrees at setup)
    back_angle = angle_with_vertical(shoulder_px[0]-hip_px[0], shoulder_px[1]-hip_px[1])
    # Target 50 degrees, deviation normalized by 50 (tolerance)
    back_quality = 100 - min(100, abs(back_angle - 50) / 50 * 100)
    
    # 4. Hip-knee-ankle alignment (proportions)
    hip_knee_dist = math.hypot(hip_px[0]-knee_px[0], hip_px[1]-knee_px[1])
    knee_ankle_dist = math.hypot(knee_px[0]-ankle_px[0], knee_px[1]-ankle_px[1])
    alignment_ratio = hip_knee_dist / (knee_ankle_dist + 1e-6)
    # Target ratio 1.2, deviation normalized by 1.2 (tolerance)
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
    
    def add_metrics(self, torso, hip, knee, form_metrics):
        self.current_rep_metrics.append([torso, hip, knee, form_metrics])
    
    def get_rep_summary(self):
        if not self.current_rep_metrics:
            return None
        
        metrics = np.array([m[:3] for m in self.current_rep_metrics])
        form_metrics = [m[3] for m in self.current_rep_metrics]
        
        duration = time.time() - self.rep_start_time if self.rep_start_time else 0
        
        avg_form_quality = np.mean([fm['overall'] for fm in form_metrics])
        avg_back_quality = np.mean([fm['back_angle'] for fm in form_metrics])
        avg_alignment_quality = np.mean([fm['hip_knee_alignment'] for fm in form_metrics])
        
        return {
            'torso': (np.mean(metrics[:, 0]), np.min(metrics[:, 0]), np.max(metrics[:, 0])),
            'hip': (np.mean(metrics[:, 1]), np.min(metrics[:, 1]), np.max(metrics[:, 1])),
            'knee': (np.mean(metrics[:, 2]), np.min(metrics[:, 2]), np.max(metrics[:, 2])),
            'form_quality': avg_form_quality,
            'back_quality': avg_back_quality,
            'alignment_quality': avg_alignment_quality,
            'duration': duration,
            # Range of motion: max torso angle - min torso angle
            'rom_quality': np.max(metrics[:, 0]) - np.min(metrics[:, 0])
        }

# -------------------- Initialize with enhanced reference system --------------------
rep_detector = DeadliftRepDetector()
hip_trajectory = deque(maxlen=150)
blur_enabled = True
logging_active = False
show_reference = True
show_trajectory = True

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
quit_btn = Button(10 + 5*(button_width + button_spacing), button_y, button_width, button_height, "QUIT", (200, 0, 0), (255, 255, 255))

buttons = [start_btn, stop_btn, blur_btn, ref_btn, traj_btn, quit_btn]

# -------------------- Mouse Callback Definition (Moved UP) --------------------
mouse_x, mouse_y = 0, 0
mouse_clicked = False

def mouse_callback(event, x, y, flags, param):
    """Handles mouse events for button clicks and hover."""
    global mouse_x, mouse_y, mouse_clicked
    mouse_x, mouse_y = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_clicked = True

# -------------------- Window Initialization (Uses mouse_callback) --------------------
cv2.namedWindow("Deadlift Form Tracker")
cv2.setMouseCallback("Deadlift Form Tracker", mouse_callback)

print("\n" + "="*60)
print("ENHANCED DEADLIFT FORM TRACKER - Plus Sign Reference System")
print("="*60)
print("Reference System Features:")
print("  ⊕ Vertical line: Ideal bar path over mid-foot")
print("  ⊕ Horizontal line: Optimal hip height")
print("  ⊕ Plus sign center: Target hip position")
print("  ⊕ Real-time form quality scoring")
print("="*60)

# -------------------- Main loop with enhanced reference system --------------------
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        if working_camera is not None:
             # Camera disconnected or error
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
    
    if pose_results.pose_landmarks:
        lm = pose_results.pose_landmarks.landmark
        
        # Determine which side (left/right) is more visible for side-view tracking
        # This is crucial for a deadlift-focused tracker assumed to be side-view
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

            # Calculate enhanced reference points
            reference_points = calculate_deadlift_reference_points(
                ankle_px, knee_px, shoulder_px, hip_px, h
            )
            
            # Calculate comprehensive form quality
            form_quality = calculate_form_quality(
                hip_px, knee_px, shoulder_px, ankle_px, reference_points
            )

            # Calculate angles
            # Torso angle: Angle of hip-shoulder line with the vertical axis
            torso_angle = angle_with_vertical(shoulder_px[0]-hip_px[0], shoulder_px[1]-hip_px[1])
            # Hip angle: Angle at the hip joint (shoulder-hip-knee)
            hip_angle = angle_between_points(shoulder_px, hip_px, knee_px)
            # Knee angle: Angle at the knee joint (hip-knee-ankle)
            knee_angle = angle_between_points(hip_px, knee_px, ankle_px)

            # Track hip trajectory
            hip_trajectory.append(hip_px)

            # Rep detection and logging
            if logging_active:
                # Add current frame's metrics
                rep_detector.add_metrics(torso_angle, hip_angle, knee_angle, form_quality)
                
                # Check for rep state change
                rep_completed, current_state = rep_detector.update(
                    lm[hip_idx].y, torso_angle, hip_angle, knee_angle, form_quality
                )
                
                if rep_completed:
                    summary = rep_detector.get_rep_summary()
                    if summary:
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
                            form_quality['hip_vertical'], form_quality['hip_height'], # Using final frame's alignment as a snapshot
                            abs(hip_px[0] - reference_points['vertical_line_x']), # Final hip displacement from bar path
                            summary['rom_quality'],
                            1.0 / summary['duration'] if summary['duration'] > 0 else 0,
                            summary['back_quality'],
                            summary['alignment_quality']
                        ])

            # Draw skeleton
            for canvas in [frame, black_canvas]:
                # Draw connections and landmarks (thick gray base, thin colored overlay)
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

            # Draw hip trajectory
            if show_trajectory and len(hip_trajectory) > 1:
                for canvas in [frame, black_canvas]:
                    points = np.array(list(hip_trajectory), dtype=np.int32)
                    cv2.polylines(canvas, [points], False, (255, 100, 255), 2)

            # Enhanced display info
            status_color = (0, 255, 0) if logging_active else (100, 100, 100)
            cv2.putText(frame, f"Reps: {rep_detector.rep_count} | State: {rep_detector.state}", 
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame, f"Form Quality: {form_quality['overall']:.0f}% | Back: {form_quality['back_angle']:.0f}%", 
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Hip Alignment: V{form_quality['hip_vertical']:.0f}% H{form_quality['hip_height']:.0f}%", 
                        (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        else:
            cv2.putText(frame, "ADJUST VIEW: Side-view body not clearly visible.", 
                        (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)


    # Draw buttons and handle clicks
    for btn in buttons:
        btn.hovered = btn.is_clicked(mouse_x, mouse_y)
        btn.draw(frame)
        btn.draw(black_canvas) # Draw on the black canvas too for visual consistency

    # Update dynamic button text
    blur_btn.text = "BLUR ON" if blur_enabled else "BLUR OFF"
    ref_btn.text = "REF ON" if show_reference else "REF OFF"
    traj_btn.text = "TRAJ ON" if show_trajectory else "TRAJ OFF"

    if mouse_clicked:
        if start_btn.is_clicked(mouse_x, mouse_y):
            if not logging_active:
                logging_active = True
                print("✓ Tracking started...")
        elif stop_btn.is_clicked(mouse_x, mouse_y):
            if logging_active:
                logging_active = False
                print("✓ Tracking stopped...")
        elif blur_btn.is_clicked(mouse_x, mouse_y):
            blur_enabled = not blur_enabled
        elif ref_btn.is_clicked(mouse_x, mouse_y):
            show_reference = not show_reference
        elif traj_btn.is_clicked(mouse_x, mouse_y):
            show_trajectory = not show_trajectory
        elif quit_btn.is_clicked(mouse_x, mouse_y):
            break
        mouse_clicked = False

    # Combine the camera frame and the black canvas
    combined = np.hstack((frame, black_canvas))
    cv2.imshow("Deadlift Form Tracker", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save CSV
if rep_data:
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_fields)
        writer.writerows(rep_data)

    print(f"\n✓ Rep data logged. CSV saved: {csv_file}")
else:
    print("\nℹ️ No reps logged. CSV file not created.")

# Cleanup
cap.release()
cv2.destroyAllWindows()
pose.close()
segmentation.close()
print("\n✓ Enhanced session complete!")