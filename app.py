import cv2
import mediapipe as mp
import numpy as np
import time
import math
import base64
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import threading
from collections import deque
import json

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

app = Flask(__name__)
CORS(app)

# Global variables for pose detection
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# Deadlift analysis classes
class DeadliftRepDetector:
    def __init__(self):
        self.state = "STANDING"
        self.rep_count = 0
        self.rep_start_time = None
        self.standing_threshold = 160
        self.bottom_threshold = 80
        self.hip_height_threshold = 0.005
        self.last_hip_y = None
        self.current_rep_metrics = []
        
    def update(self, hip_y_norm, torso_angle, hip_angle, knee_angle):
        rep_completed = False
        
        if self.last_hip_y is None:
            self.last_hip_y = hip_y_norm
            return rep_completed, self.state
        
        hip_moving_down = hip_y_norm > self.last_hip_y + self.hip_height_threshold
        hip_moving_up = hip_y_norm < self.last_hip_y - self.hip_height_threshold
        
        if self.state == "STANDING":
            if torso_angle < self.standing_threshold - 10 and hip_moving_down:
                self.state = "DESCENDING"
                self.rep_start_time = time.time()
                self.current_rep_metrics = []
                
        elif self.state == "DESCENDING":
            if torso_angle < self.bottom_threshold and hip_angle < 110:
                self.state = "BOTTOM"
                
        elif self.state == "BOTTOM":
            if hip_moving_up and torso_angle > self.bottom_threshold + 5:
                self.state = "ASCENDING"
                
        elif self.state == "ASCENDING":
            if torso_angle > self.standing_threshold - 5 and hip_angle > 165:
                self.state = "STANDING"
                self.rep_count += 1
                rep_completed = True
        
        self.last_hip_y = hip_y_norm
        return rep_completed, self.state

class BiomechanicalAnalyzer:
    def __init__(self):
        self.body_weight = 75
        self.barbell_weight = 60
        
    def calculate_spinal_loading(self, landmarks, width, height):
        hip_px = self.lm_to_pixel(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], width, height)
        shoulder_px = self.lm_to_pixel(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], width, height)
        
        back_angle = math.atan2(shoulder_px[0]-hip_px[0], shoulder_px[1]-hip_px[1])
        compression_force = self.barbell_weight * (1 + abs(math.sin(back_angle)))
        return compression_force
    
    def lm_to_pixel(self, landmark, width, height):
        return int(landmark.x * width), int(landmark.y * height)

# Helper functions
def angle_with_vertical(vx, vy):
    dot = vx * 0 + vy * (-1)
    mag_v = math.hypot(vx, vy)
    if mag_v == 0:
        return 0.0
    cos_a = max(min(dot / mag_v, 1), -1)
    return math.degrees(math.acos(cos_a))

def angle_between_points(a, b, c):
    ba = np.array([a[0]-b[0], a[1]-b[1]])
    bc = np.array([c[0]-b[0], c[1]-b[1]])
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    cos_angle = max(min(cos_angle,1),-1)
    return math.degrees(np.arccos(cos_angle))

def calculate_deadlift_reference_points(ankle_px, knee_px, shoulder_px, hip_px, frame_height):
    mid_foot_x = ankle_px[0]
    knee_height = knee_px[1]
    shoulder_height = shoulder_px[1]
    optimal_hip_height = knee_height + (shoulder_height - knee_height) * 0.6
    
    return {
        'vertical_line_x': mid_foot_x,
        'optimal_hip_height': optimal_hip_height,
        'mid_foot': (mid_foot_x, ankle_px[1])
    }

def draw_plus_sign_reference(img, reference_points, size=150, color=(0, 255, 255), thickness=3):
    vx = reference_points['vertical_line_x']
    hip_y = int(reference_points['optimal_hip_height'])
    
    # Vertical reference line
    cv2.line(img, (vx, 0), (vx, img.shape[0]), color, thickness-1)
    
    # Horizontal reference line
    cv2.line(img, (vx-size, hip_y), (vx+size, hip_y), color, thickness-1)
    
    # Plus sign center
    center_x, center_y = vx, hip_y
    arm_length = size // 2
    
    # Enhanced plus sign
    cv2.line(img, (center_x, center_y - arm_length), (center_x, center_y + arm_length), 
             (255, 255, 255), thickness+1)
    cv2.line(img, (center_x - arm_length, center_y), (center_x + arm_length, center_y), 
             (255, 255, 255), thickness+1)
    cv2.circle(img, (center_x, center_y), arm_length, color, thickness)
    cv2.circle(img, (center_x, center_y), 15, (0, 0, 255), -1)
    
    # Labels
    cv2.putText(img, "IDEAL HIP HEIGHT", (center_x + 40, center_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(img, "BAR PATH", (center_x + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Mid-foot marker
    cv2.circle(img, reference_points['mid_foot'], 8, (0, 255, 0), -1)
    cv2.putText(img, "MID-FOOT", (reference_points['mid_foot'][0] - 40, reference_points['mid_foot'][1] + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

def draw_form_feedback(img, hip_px, form_quality, reference_points):
    vx = reference_points['vertical_line_x']
    optimal_y = int(reference_points['optimal_hip_height'])
    
    # Draw current hip position
    cv2.circle(img, hip_px, 12, (0, 0, 255), -1)
    cv2.circle(img, hip_px, 12, (255, 255, 255), 2)
    
    # Connection lines to reference
    cv2.line(img, hip_px, (vx, hip_px[1]), (255, 100, 100), 2)
    cv2.line(img, hip_px, (hip_px[0], optimal_y), (255, 100, 100), 2)
    
    # Quality indicator
    if form_quality > 80:
        quality_color = (0, 255, 0)
    elif form_quality > 60:
        quality_color = (0, 255, 255)
    else:
        quality_color = (0, 0, 255)
    
    # Quality bar
    bar_x, bar_y = 20, img.shape[0] - 100
    bar_width = 200
    bar_height = 20
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + int(bar_width * form_quality / 100), bar_y + bar_height), quality_color, -1)
    cv2.putText(img, f"FORM: {form_quality:.0f}%", (bar_x, bar_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, quality_color, 2)

def calculate_form_quality(hip_px, knee_px, shoulder_px, ankle_px, reference_points):
    vx = reference_points['vertical_line_x']
    optimal_hip_y = reference_points['optimal_hip_height']
    
    # Hip alignment quality
    hip_v_alignment = 100 - min(100, (abs(hip_px[0] - vx) / (vx * 0.2 + 1e-6)) * 100)
    
    # Hip height quality
    hip_h_alignment = 100 - min(100, (abs(hip_px[1] - optimal_hip_y) / (optimal_hip_y * 0.3 + 1e-6)) * 100)
    
    # Back angle quality
    back_angle = angle_with_vertical(shoulder_px[0]-hip_px[0], shoulder_px[1]-hip_px[1])
    back_quality = 100 - min(100, abs(back_angle - 50) / 50 * 100)
    
    # Hip-knee-ankle alignment
    hip_knee_dist = math.hypot(hip_px[0]-knee_px[0], hip_px[1]-knee_px[1])
    knee_ankle_dist = math.hypot(knee_px[0]-ankle_px[0], knee_px[1]-ankle_px[1])
    alignment_ratio = hip_knee_dist / (knee_ankle_dist + 1e-6)
    alignment_quality = 100 - min(100, abs(alignment_ratio - 1.2) / 1.2 * 100)
    
    overall_quality = (hip_v_alignment + hip_h_alignment + back_quality + alignment_quality) / 4
    return overall_quality

def create_mirrored_landmarks(original_landmarks):
    """Create a list of mirrored landmarks by flipping x coordinates"""
    mirrored_landmarks = []
    for landmark in original_landmarks.landmark:
        # Create a new landmark with mirrored x coordinate
        mirrored_landmark = type(landmark)()
        mirrored_landmark.x = 1.0 - landmark.x  # Mirror horizontally
        mirrored_landmark.y = landmark.y
        mirrored_landmark.z = landmark.z
        mirrored_landmark.visibility = landmark.visibility
        mirrored_landmarks.append(mirrored_landmark)
    return mirrored_landmarks

def draw_mirrored_pose_landmarks(image, pose_landmarks):
    """Draw pose landmarks with mirrored x coordinates"""
    if pose_landmarks is None:
        return
    
    # Create mirrored landmarks
    mirrored_landmarks = create_mirrored_landmarks(pose_landmarks)
    
    # Draw connections
    for connection in mp_pose.POSE_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]
        
        start_lm = mirrored_landmarks[start_idx]
        end_lm = mirrored_landmarks[end_idx]
        
        # Convert to pixel coordinates
        h, w, _ = image.shape
        start_px = (int(start_lm.x * w), int(start_lm.y * h))
        end_px = (int(end_lm.x * w), int(end_lm.y * h))
        
        # Draw connection line
        cv2.line(image, start_px, end_px, (0, 200, 255), 3)
    
    # Draw landmarks
    for landmark in mirrored_landmarks:
        h, w, _ = image.shape
        px = (int(landmark.x * w), int(landmark.y * h))
        
        # Draw landmark point
        cv2.circle(image, px, 4, (0, 255, 0), -1)
        cv2.circle(image, px, 4, (255, 255, 255), 1)

# Initialize detectors
rep_detector = DeadliftRepDetector()
biomech_analyzer = BiomechanicalAnalyzer()

# Global trajectory storage
trajectories = {
    'hip': deque(maxlen=50),
    'shoulder': deque(maxlen=50),
    'knee': deque(maxlen=50),
    'ankle': deque(maxlen=50)
}

def generate_black_screen_with_pose():
    """Generate frames with pose mesh on black background - FIXED MIRRORING"""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Create black screen
        black_screen = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Process frame with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_frame)
        
        if pose_results.pose_landmarks:
            # Draw mirrored pose landmarks using our custom function
            draw_mirrored_pose_landmarks(black_screen, pose_results.pose_landmarks)
            
            # Create mirrored landmarks for analysis
            mirrored_landmarks = create_mirrored_landmarks(pose_results.pose_landmarks)
            
            # Use left or right side based on visibility (now properly oriented)
            left_hip_vis = mirrored_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility
            right_hip_vis = mirrored_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility
            use_left = left_hip_vis >= right_hip_vis and left_hip_vis > 0.5
            
            if use_left:
                hip_idx = mp_pose.PoseLandmark.LEFT_HIP.value
                knee_idx = mp_pose.PoseLandmark.LEFT_KNEE.value
                ankle_idx = mp_pose.PoseLandmark.LEFT_ANKLE.value
                shoulder_idx = mp_pose.PoseLandmark.LEFT_SHOULDER.value
            else:
                hip_idx = mp_pose.PoseLandmark.RIGHT_HIP.value
                knee_idx = mp_pose.PoseLandmark.RIGHT_KNEE.value
                ankle_idx = mp_pose.PoseLandmark.RIGHT_ANKLE.value
                shoulder_idx = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            
            # Convert to pixel coordinates
            def lm_to_pixel(landmark, width, height):
                return int(landmark.x * width), int(landmark.y * height)
            
            hip_px = lm_to_pixel(mirrored_landmarks[hip_idx], 640, 480)
            knee_px = lm_to_pixel(mirrored_landmarks[knee_idx], 640, 480)
            ankle_px = lm_to_pixel(mirrored_landmarks[ankle_idx], 640, 480)
            shoulder_px = lm_to_pixel(mirrored_landmarks[shoulder_idx], 640, 480)
            
            # Update trajectories
            trajectories['hip'].append(hip_px)
            trajectories['shoulder'].append(shoulder_px)
            trajectories['knee'].append(knee_px)
            trajectories['ankle'].append(ankle_px)
            
            # Calculate reference points
            reference_points = calculate_deadlift_reference_points(
                ankle_px, knee_px, shoulder_px, hip_px, 480
            )
            
            # Calculate form quality
            form_quality = calculate_form_quality(
                hip_px, knee_px, shoulder_px, ankle_px, reference_points
            )
            
            # Calculate angles for rep detection
            torso_angle = angle_with_vertical(shoulder_px[0]-hip_px[0], shoulder_px[1]-hip_px[1])
            hip_angle = angle_between_points(shoulder_px, hip_px, knee_px)
            knee_angle = angle_between_points(hip_px, knee_px, ankle_px)
            
            # Update rep detector
            rep_completed, current_state = rep_detector.update(
                mirrored_landmarks[hip_idx].y, torso_angle, hip_angle, knee_angle
            )
            
            # Draw reference system
            draw_plus_sign_reference(black_screen, reference_points)
            
            # Draw form feedback
            draw_form_feedback(black_screen, hip_px, form_quality, reference_points)
            
            # Draw trajectories
            colors = {'hip': (0, 255, 255), 'shoulder': (255, 255, 0), 
                     'knee': (255, 0, 255), 'ankle': (0, 255, 0)}
            
            for landmark_name, points in trajectories.items():
                if len(points) > 1:
                    color = colors.get(landmark_name, (255, 255, 255))
                    for i in range(1, len(points)):
                        cv2.line(black_screen, points[i-1], points[i], color, 2)
                    if points:
                        cv2.circle(black_screen, points[-1], 8, color, -1)
            
            # Display rep count and state
            cv2.putText(black_screen, f"Reps: {rep_detector.rep_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(black_screen, f"State: {rep_detector.state}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(black_screen, f"Form: {form_quality:.0f}%", 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Calculate and display biomechanical data
            spinal_load = biomech_analyzer.calculate_spinal_loading(mirrored_landmarks, 640, 480)
            cv2.putText(black_screen, f"Spinal Load: {spinal_load:.1f}N", 
                       (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', black_screen)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return """
    <html>
        <head>
            <title>Deadlift Analysis API</title>
            <style>
                body { margin: 0; padding: 20px; background: #1a1a1a; color: white; font-family: Arial; }
                .container { max-width: 800px; margin: 0 auto; text-align: center; }
                img { width: 100%; max-width: 640px; border: 2px solid #333; border-radius: 10px; }
                .stats { background: #2a2a2a; padding: 20px; border-radius: 10px; margin: 20px 0; }
                .endpoints { text-align: left; background: #2a2a2a; padding: 20px; border-radius: 10px; margin: 20px 0; }
                code { background: #1a1a1a; padding: 2px 6px; border-radius: 4px; font-family: monospace; }
                .status-fixed { color: #4CAF50; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🏋️ Deadlift Form Analysis API</h1>
                <p>Real-time pose detection with biomechanical analysis on black screen</p>
                <p class="status-fixed">✅ FIXED: Pose direction now matches your actual movement</p>
                
                <div class="stats">
                    <h3>Live Pose Detection Feed</h3>
                    <img src="/video_feed" alt="Live Pose Detection">
                </div>
                
                <div class="endpoints">
                    <h3>📡 API Endpoints</h3>
                    <p><strong>GET <code>/video_feed</code></strong> - Live video stream with pose mesh on black screen</p>
                    <p><strong>GET <code>/stats</code></strong> - Current statistics (reps, form quality, etc.)</p>
                    <p><strong>POST <code>/reset</code></strong> - Reset rep counter</p>
                    <p><strong>POST <code>/process_frame</code></strong> - Process single frame</p>
                </div>
                
                <div class="stats">
                    <h3>📱 Mobile Integration</h3>
                    <p>Your Android app can directly consume the <code>/video_feed</code> endpoint</p>
                    <p>Simply point your ImageView to: <code>http://your-server:5005/video_feed</code></p>
                </div>
            </div>
        </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    """Video streaming route that shows pose mesh on black screen"""
    return Response(generate_black_screen_with_pose(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    """Get current statistics"""
    stats = {
        'reps': rep_detector.rep_count,
        'state': rep_detector.state,
        'timestamp': time.time()
    }
    return jsonify(stats)

@app.route('/reset', methods=['POST'])
def reset_counter():
    """Reset rep counter"""
    rep_detector.rep_count = 0
    rep_detector.state = "STANDING"
    rep_detector.last_hip_y = None
    
    # Clear trajectories
    for key in trajectories:
        trajectories[key].clear()
    
    return jsonify({'message': 'Counter reset', 'reps': 0})

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process a single frame and return analysis results"""
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'}), 400
    
    file = request.files['frame']
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return jsonify({'error': 'Could not decode image'}), 400
    
    # Process frame (similar to video feed but return JSON)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb_frame)
    
    analysis = {
        'pose_detected': pose_results.pose_landmarks is not None,
        'reps': rep_detector.rep_count,
        'state': rep_detector.state
    }
    
    return jsonify(analysis)

if __name__ == '__main__':
    print("Starting Deadlift Analysis API on port 5005...")
    print("Access the live feed at: http://localhost:5005/")
    print("Video feed endpoint: http://localhost:5005/video_feed")
    print("✅ Mirroring fix applied - Pose direction now matches your movement")
    app.run(host='0.0.0.0', port=5005, debug=False, threaded=True)


