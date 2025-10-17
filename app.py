# from flask import Flask, render_template, Response, jsonify, request
# from flask_cors import CORS
# import cv2
# import mediapipe as mp
# import numpy as np
# import time
# import math
# import json
# from collections import deque
# import base64

# app = Flask(__name__)
# CORS(app)

# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils
# mp_selfie_segmentation = mp.solutions.selfie_segmentation

# pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
# segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# class DeadliftRepDetector:
#     def __init__(self):
#         self.state = "STANDING"
#         self.rep_count = 0
#         self.current_rep_metrics = []
#         self.rep_start_time = None
#         self.standing_threshold = 160
#         self.bottom_threshold = 80
#         self.hip_height_threshold = 0.005
#         self.last_hip_y = None

#     def update(self, hip_y_norm, torso_angle, hip_angle, knee_angle, form_quality):
#         rep_completed = False

#         if self.last_hip_y is None:
#             self.last_hip_y = hip_y_norm
#             return rep_completed, self.state

#         hip_moving_down = hip_y_norm > self.last_hip_y + self.hip_height_threshold
#         hip_moving_up = hip_y_norm < self.last_hip_y - self.hip_height_threshold

#         if self.state == "STANDING":
#             if torso_angle < self.standing_threshold - 10 and hip_moving_down:
#                 self.state = "DESCENDING"
#                 self.rep_start_time = time.time()
#                 self.current_rep_metrics = []

#         elif self.state == "DESCENDING":
#             if torso_angle < self.bottom_threshold and hip_angle < 110:
#                 self.state = "BOTTOM"

#         elif self.state == "BOTTOM":
#             if hip_moving_up and torso_angle > self.bottom_threshold + 5:
#                 self.state = "ASCENDING"

#         elif self.state == "ASCENDING":
#             if torso_angle > self.standing_threshold - 5 and hip_angle > 165:
#                 self.state = "STANDING"
#                 self.rep_count += 1
#                 rep_completed = True

#         self.last_hip_y = hip_y_norm
#         return rep_completed, self.state

#     def add_metrics(self, torso, hip, knee, form_quality):
#         self.current_rep_metrics.append([torso, hip, knee, form_quality])

#     def reset(self):
#         self.state = "STANDING"
#         self.rep_count = 0
#         self.current_rep_metrics = []
#         self.rep_start_time = None
#         self.last_hip_y = None

# rep_detector = DeadliftRepDetector()
# tracking_active = False
# camera = None

# def lm_to_pixel(landmark, width, height):
#     return int(landmark.x * width), int(landmark.y * height)

# def angle_with_vertical(vx, vy):
#     dot = vx * 0 + vy * (-1)
#     mag_v = math.hypot(vx, vy)
#     if mag_v == 0:
#         return 0.0
#     cos_a = max(min(dot / mag_v, 1), -1)
#     return math.degrees(math.acos(cos_a))

# def angle_between_points(a, b, c):
#     ba = np.array([a[0]-b[0], a[1]-b[1]])
#     bc = np.array([c[0]-b[0], c[1]-b[1]])
#     cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
#     cos_angle = max(min(cos_angle,1),-1)
#     return math.degrees(np.arccos(cos_angle))

# def calculate_form_quality(hip_px, knee_px, shoulder_px, ankle_px, width, height):
#     vx = width // 2
#     optimal_hip_y = knee_px[1] + (shoulder_px[1] - knee_px[1]) * 0.6

#     hip_v_alignment = 100 - min(100, (abs(hip_px[0] - vx) / (vx * 0.2 + 1e-6)) * 100)
#     hip_h_alignment = 100 - min(100, (abs(hip_px[1] - optimal_hip_y) / (optimal_hip_y * 0.3 + 1e-6)) * 100)

#     back_angle = angle_with_vertical(shoulder_px[0]-hip_px[0], shoulder_px[1]-hip_px[1])
#     back_quality = 100 - min(100, abs(back_angle - 50) / 50 * 100)

#     overall_quality = (hip_v_alignment + hip_h_alignment + back_quality) / 3

#     return {
#         'overall': overall_quality,
#         'hip_vertical': hip_v_alignment,
#         'hip_height': hip_h_alignment,
#         'back_angle': back_quality
#     }

# def generate_frames():
#     global camera, tracking_active, rep_detector

#     if camera is None:
#         working_camera = None
#         for index in range(10):
#             temp_cap = cv2.VideoCapture(index)
#             time.sleep(0.3)
#             if temp_cap.isOpened():
#                 ret, frame = temp_cap.read()
#                 if ret and frame is not None:
#                     working_camera = index
#                     temp_cap.release()
#                     break
#             temp_cap.release()

#         if working_camera is None:
#             return

#         camera = cv2.VideoCapture(working_camera)
#         camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#     while True:
#         success, frame = camera.read()
#         if not success:
#             break

#         frame = cv2.flip(frame, 1)
#         h, w, _ = frame.shape

#         black_canvas = np.zeros((h, w, 3), dtype=np.uint8)

#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         pose_results = pose.process(rgb_frame)

#         metrics_data = {
#             'reps': rep_detector.rep_count,
#             'state': rep_detector.state,
#             'form_quality': 0,
#             'torso_angle': 0,
#             'hip_angle': 0,
#             'knee_angle': 0
#         }

#         if pose_results.pose_landmarks:
#             lm = pose_results.pose_landmarks.landmark

#             left_hip_vis = lm[mp_pose.PoseLandmark.LEFT_HIP].visibility
#             right_hip_vis = lm[mp_pose.PoseLandmark.RIGHT_HIP].visibility
#             use_left = left_hip_vis >= right_hip_vis and left_hip_vis > 0.5
#             use_right = right_hip_vis > left_hip_vis and right_hip_vis > 0.5

#             if use_left or use_right:
#                 if use_left:
#                     hip_idx, knee_idx, ankle_idx, shoulder_idx = (
#                         mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE,
#                         mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_SHOULDER
#                     )
#                 else:
#                     hip_idx, knee_idx, ankle_idx, shoulder_idx = (
#                         mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE,
#                         mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_SHOULDER
#                     )

#                 hip_px = lm_to_pixel(lm[hip_idx], w, h)
#                 knee_px = lm_to_pixel(lm[knee_idx], w, h)
#                 ankle_px = lm_to_pixel(lm[ankle_idx], w, h)
#                 shoulder_px = lm_to_pixel(lm[shoulder_idx], w, h)

#                 torso_angle = angle_with_vertical(shoulder_px[0]-hip_px[0], shoulder_px[1]-hip_px[1])
#                 hip_angle = angle_between_points(shoulder_px, hip_px, knee_px)
#                 knee_angle = angle_between_points(hip_px, knee_px, ankle_px)

#                 form_quality = calculate_form_quality(hip_px, knee_px, shoulder_px, ankle_px, w, h)

#                 metrics_data.update({
#                     'form_quality': round(form_quality['overall'], 1),
#                     'torso_angle': round(torso_angle, 1),
#                     'hip_angle': round(hip_angle, 1),
#                     'knee_angle': round(knee_angle, 1)
#                 })

#                 if tracking_active:
#                     rep_detector.add_metrics(torso_angle, hip_angle, knee_angle, form_quality)
#                     rep_completed, current_state = rep_detector.update(
#                         lm[hip_idx].y, torso_angle, hip_angle, knee_angle, form_quality
#                     )
#                     metrics_data['reps'] = rep_detector.rep_count
#                     metrics_data['state'] = rep_detector.state

#                 mp_drawing.draw_landmarks(
#                     black_canvas, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                     landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
#                     connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=3)
#                 )

#                 vx = w // 2
#                 optimal_hip_y = int(knee_px[1] + (shoulder_px[1] - knee_px[1]) * 0.6)

#                 cv2.line(black_canvas, (vx, 0), (vx, h), (0, 255, 0), 1)
#                 cv2.line(black_canvas, (vx-100, optimal_hip_y), (vx+100, optimal_hip_y), (0, 255, 0), 1)

#                 cv2.circle(black_canvas, (vx, optimal_hip_y), 50, (0, 255, 0), 2)
#                 cv2.circle(black_canvas, (vx, optimal_hip_y), 10, (0, 255, 0), -1)

#                 cv2.circle(black_canvas, hip_px, 8, (0, 255, 255), -1)

#                 quality_color = (0, 255, 0) if form_quality['overall'] > 80 else (0, 255, 255) if form_quality['overall'] > 60 else (0, 0, 255)

#                 cv2.putText(black_canvas, f"Reps: {rep_detector.rep_count} | State: {rep_detector.state}",
#                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#                 cv2.putText(black_canvas, f"Form: {form_quality['overall']:.0f}%",
#                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)
#                 cv2.putText(black_canvas, f"Torso: {torso_angle:.0f}deg | Hip: {hip_angle:.0f}deg | Knee: {knee_angle:.0f}deg",
#                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

#         ret, buffer = cv2.imencode('.jpg', black_canvas)
#         frame_bytes = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/start', methods=['POST'])
# def start_tracking():
#     global tracking_active
#     tracking_active = True
#     return jsonify({'status': 'started', 'message': 'Tracking started'})

# @app.route('/stop', methods=['POST'])
# def stop_tracking():
#     global tracking_active
#     tracking_active = False
#     return jsonify({
#         'status': 'stopped',
#         'message': 'Tracking stopped',
#         'total_reps': rep_detector.rep_count
#     })

# @app.route('/reset', methods=['POST'])
# def reset_tracking():
#     global tracking_active
#     tracking_active = False
#     rep_detector.reset()
#     return jsonify({'status': 'reset', 'message': 'Tracker reset'})

# @app.route('/metrics', methods=['GET'])
# def get_metrics():
#     return jsonify({
#         'reps': rep_detector.rep_count,
#         'state': rep_detector.state,
#         'tracking_active': tracking_active
#     })

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5005, debug=False)


from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import time
import math
import csv
import base64
from collections import deque
import threading
import os

app = Flask(__name__)

# Global variables for tracking state
tracking_active = False
current_frame = None
frame_lock = threading.Lock()
rep_data = {
    'rep_count': 0,
    'form_quality': 0,
    'current_state': "STANDING",
    'torso_angle': 0,
    'hip_angle': 0,
    'knee_angle': 0,
    'hip_alignment': 0,
    'back_quality': 0
}
data_lock = threading.Lock()

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)

# Helper functions
def lm_to_pixel(landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)

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

def calculate_form_quality(hip_px, knee_px, shoulder_px, ankle_px, reference_points):
    vx = reference_points['vertical_line_x']
    optimal_hip_y = reference_points['optimal_hip_height']
    
    hip_v_alignment = 100 - min(100, (abs(hip_px[0] - vx) / (vx * 0.2 + 1e-6)) * 100)
    hip_h_alignment = 100 - min(100, (abs(hip_px[1] - optimal_hip_y) / (optimal_hip_y * 0.3 + 1e-6)) * 100)
    
    back_angle = angle_with_vertical(shoulder_px[0]-hip_px[0], shoulder_px[1]-hip_px[1])
    back_quality = 100 - min(100, abs(back_angle - 50) / 50 * 100)
    
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

class DeadliftRepDetector:
    def __init__(self):
        self.state = "STANDING"
        self.rep_count = 0
        self.standing_threshold = 160
        self.bottom_threshold = 80
        self.hip_height_threshold = 0.005
        self.last_hip_y = None
        
    def update(self, hip_y_norm, torso_angle, hip_angle, knee_angle, form_quality):
        rep_completed = False
        
        if self.last_hip_y is None:
            self.last_hip_y = hip_y_norm
            return rep_completed, self.state
        
        hip_moving_down = hip_y_norm > self.last_hip_y + self.hip_height_threshold
        hip_moving_up = hip_y_norm < self.last_hip_y - self.hip_height_threshold
        
        if self.state == "STANDING":
            if torso_angle < self.standing_threshold - 10 and hip_moving_down:
                self.state = "DESCENDING"
                
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

rep_detector = DeadliftRepDetector()

def process_frame(frame_data):
    global tracking_active, current_frame, rep_data
    
    if not tracking_active:
        return
    
    try:
        # Decode base64 image
        header, encoded = frame_data.split(",", 1)
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return
        
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        pose_results = pose.process(rgb_frame)
        
        if pose_results.pose_landmarks:
            lm = pose_results.pose_landmarks.landmark
            
            # Determine side
            left_hip_vis = lm[mp_pose.PoseLandmark.LEFT_HIP].visibility
            right_hip_vis = lm[mp_pose.PoseLandmark.RIGHT_HIP].visibility
            use_left = left_hip_vis >= right_hip_vis and left_hip_vis > 0.5
            use_right = right_hip_vis > left_hip_vis and right_hip_vis > 0.5
            
            if use_left or use_right:
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

                # Calculate reference points and form quality
                reference_points = calculate_deadlift_reference_points(
                    ankle_px, knee_px, shoulder_px, hip_px, h
                )
                
                form_quality = calculate_form_quality(
                    hip_px, knee_px, shoulder_px, ankle_px, reference_points
                )

                # Calculate angles
                torso_angle = angle_with_vertical(shoulder_px[0]-hip_px[0], shoulder_px[1]-hip_px[1])
                hip_angle = angle_between_points(shoulder_px, hip_px, knee_px)
                knee_angle = angle_between_points(hip_px, knee_px, ankle_px)

                # Update rep detection
                rep_completed, current_state = rep_detector.update(
                    lm[hip_idx].y, torso_angle, hip_angle, knee_angle, form_quality
                )

                # Update global data
                with data_lock:
                    rep_data.update({
                        'rep_count': rep_detector.rep_count,
                        'form_quality': form_quality['overall'],
                        'current_state': current_state,
                        'torso_angle': torso_angle,
                        'hip_angle': hip_angle,
                        'knee_angle': knee_angle,
                        'hip_alignment': form_quality['hip_vertical'],
                        'back_quality': form_quality['back_angle']
                    })

                # Draw skeleton and landmarks on frame
                mp_drawing.draw_landmarks(
                    frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,200,255), thickness=3)
                )

                # Draw reference system
                vx = reference_points['vertical_line_x']
                hip_y = int(reference_points['optimal_hip_height'])
                
                # Vertical reference line
                cv2.line(frame, (vx, 0), (vx, h), (0, 255, 255), 2)
                # Horizontal reference line
                cv2.line(frame, (vx-100, hip_y), (vx+100, hip_y), (0, 255, 255), 2)
                
                # Plus sign
                cv2.line(frame, (vx, hip_y-50), (vx, hip_y+50), (255, 255, 255), 3)
                cv2.line(frame, (vx-50, hip_y), (vx+50, hip_y), (255, 255, 255), 3)
                
                # Current hip position
                cv2.circle(frame, hip_px, 10, (0, 0, 255), -1)

        # Encode processed frame back to base64
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        processed_frame = base64.b64encode(buffer).decode('utf-8')
        
        with frame_lock:
            current_frame = f"data:image/jpeg;base64,{processed_frame}"
            
    except Exception as e:
        print(f"Error processing frame: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_tracking', methods=['POST'])
def start_tracking():
    global tracking_active
    tracking_active = True
    return jsonify({'status': 'started'})

@app.route('/stop_tracking', methods=['POST'])
def stop_tracking():
    global tracking_active
    tracking_active = False
    return jsonify({'status': 'stopped'})

@app.route('/get_data', methods=['GET'])
def get_data():
    with data_lock:
        return jsonify(rep_data)

@app.route('/process_frame', methods=['POST'])
def process_frame_route():
    frame_data = request.json.get('frame')
    if frame_data:
        # Process in a separate thread to avoid blocking
        threading.Thread(target=process_frame, args=(frame_data,)).start()
        return jsonify({'status': 'processing'})
    return jsonify({'error': 'No frame data'})

@app.route('/get_processed_frame', methods=['GET'])
def get_processed_frame():
    with frame_lock:
        if current_frame:
            return jsonify({'frame': current_frame})
    return jsonify({'frame': None})

@app.route('/reset', methods=['POST'])
def reset():
    global rep_detector, rep_data
    rep_detector = DeadliftRepDetector()
    with data_lock:
        rep_data.update({
            'rep_count': 0,
            'form_quality': 0,
            'current_state': "STANDING",
            'torso_angle': 0,
            'hip_angle': 0,
            'knee_angle': 0,
            'hip_alignment': 0,
            'back_quality': 0
        })
    return jsonify({'status': 'reset'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 50005)
    app.run(host='0.0.0.0', port=port, debug=False)