import cv2
import mediapipe as mp
import numpy as np
import time
import math
import base64
import json
import logging
import os
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

app = Flask(__name__)
CORS(app)

class DeadliftAnalysis:
    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
            static_image_mode=False
        )
        self.rep_detector = DeadliftRepDetector()
        self.last_analysis_time = time.time()
        self.frame_count = 0
        self.analysis_history = deque(maxlen=100)

    def process_frame(self, frame):
        try:
            self.frame_count += 1
            
            # Resize for consistent processing
            frame = cv2.resize(frame, (640, 480))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            pose_results = self.pose.process(rgb_frame)
            
            analysis_result = {
                'pose_detected': False,
                'reps': self.rep_detector.rep_count,
                'state': self.rep_detector.state,
                'form_quality': 0.0,
                'spinal_load': 0.0,
                'confidence': 0.0,
                'landmarks': [],
                'timestamp': time.time(),
                'frame_count': self.frame_count
            }
            
            annotated_frame = frame.copy()
            
            if pose_results.pose_landmarks:
                analysis_result['pose_detected'] = True
                landmarks = pose_results.pose_landmarks.landmark
                
                # Draw pose landmarks
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
                
                # Extract key landmarks
                key_points = {}
                for landmark in [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE,
                               mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_SHOULDER,
                               mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE,
                               mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_SHOULDER]:
                    lm = landmarks[landmark.value]
                    key_points[landmark.name] = {
                        'x': lm.x, 
                        'y': lm.y, 
                        'z': lm.z, 
                        'visibility': lm.visibility
                    }
                
                analysis_result['landmarks'] = key_points
                
                # Calculate metrics
                if (mp_pose.PoseLandmark.RIGHT_HIP.name in key_points and 
                    mp_pose.PoseLandmark.RIGHT_SHOULDER.name in key_points):
                    
                    hip = key_points[mp_pose.PoseLandmark.RIGHT_HIP.name]
                    shoulder = key_points[mp_pose.PoseLandmark.RIGHT_SHOULDER.name]
                    knee = key_points.get(mp_pose.PoseLandmark.RIGHT_KNEE.name, {})
                    ankle = key_points.get(mp_pose.PoseLandmark.RIGHT_ANKLE.name, {})
                    
                    # Calculate angles
                    torso_angle = self.calculate_torso_angle(hip, shoulder)
                    hip_angle = self.calculate_hip_angle(hip, shoulder, knee)
                    knee_angle = self.calculate_knee_angle(hip, knee, ankle)
                    
                    # Update rep detector
                    rep_completed, current_state = self.rep_detector.update(
                        hip['y'], torso_angle, hip_angle, knee_angle
                    )
                    
                    analysis_result['state'] = current_state
                    analysis_result['reps'] = self.rep_detector.rep_count
                    
                    # Calculate form quality
                    form_quality = self.calculate_form_quality(hip, shoulder, knee, ankle)
                    analysis_result['form_quality'] = form_quality
                    
                    # Calculate spinal load
                    spinal_load = self.calculate_spinal_load(hip, shoulder)
                    analysis_result['spinal_load'] = spinal_load
                    
                    # Calculate confidence from landmark visibility
                    visibility_scores = [
                        hip.get('visibility', 0),
                        shoulder.get('visibility', 0),
                        knee.get('visibility', 0),
                        ankle.get('visibility', 0)
                    ]
                    analysis_result['confidence'] = np.mean(visibility_scores)
                    
                    # Add analysis text to annotated frame
                    self.add_analysis_text(annotated_frame, analysis_result)
            
            # Encode annotated frame
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            annotated_frame_base64 = base64.b64encode(buffer).decode('utf-8')
            analysis_result['annotated_frame'] = f"data:image/jpeg;base64,{annotated_frame_base64}"
            
            # Store in history
            self.analysis_history.append(analysis_result)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return None
    
    def calculate_torso_angle(self, hip, shoulder):
        try:
            dx = shoulder['x'] - hip['x']
            dy = shoulder['y'] - hip['y']
            return angle_with_vertical(dx, dy)
        except:
            return 0.0
    
    def calculate_hip_angle(self, hip, shoulder, knee):
        try:
            if 'x' not in knee:
                return 0.0
            a = np.array([shoulder['x'], shoulder['y']])
            b = np.array([hip['x'], hip['y']])
            c = np.array([knee['x'], knee['y']])
            return angle_between_points(a, b, c)
        except:
            return 0.0
    
    def calculate_knee_angle(self, hip, knee, ankle):
        try:
            if 'x' not in ankle:
                return 0.0
            a = np.array([hip['x'], hip['y']])
            b = np.array([knee['x'], knee['y']])
            c = np.array([ankle['x'], ankle['y']])
            return angle_between_points(a, b, c)
        except:
            return 0.0
    
    def calculate_form_quality(self, hip, shoulder, knee, ankle):
        try:
            # Simplified form quality calculation
            quality = 80.0  # Base quality
            
            # Adjust based on torso angle (ideal: 40-60 degrees)
            torso_angle = self.calculate_torso_angle(hip, shoulder)
            if 40 <= torso_angle <= 60:
                quality += 10
            else:
                quality -= abs(torso_angle - 50) * 0.5
            
            return max(0, min(100, quality))
        except:
            return 50.0
    
    def calculate_spinal_load(self, hip, shoulder):
        try:
            torso_angle = self.calculate_torso_angle(hip, shoulder)
            barbell_weight = 60  # kg
            return barbell_weight * (1 + abs(math.sin(math.radians(torso_angle))))
        except:
            return 0.0
    
    def add_analysis_text(self, frame, analysis):
        try:
            # Add analysis text to the frame
            text_color = (0, 255, 0) if analysis['form_quality'] > 70 else (0, 165, 255) if analysis['form_quality'] > 50 else (0, 0, 255)
            
            cv2.putText(frame, f"Reps: {analysis['reps']}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            cv2.putText(frame, f"State: {analysis['state']}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            cv2.putText(frame, f"Form: {analysis['form_quality']:.1f}%", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            cv2.putText(frame, f"Load: {analysis['spinal_load']:.0f}N", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            cv2.putText(frame, f"Conf: {analysis['confidence']:.1%}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        except Exception as e:
            logger.error(f"Error adding text to frame: {e}")
    
    def get_stats(self):
        return {
            'total_frames': self.frame_count,
            'total_reps': self.rep_detector.rep_count,
            'current_state': self.rep_detector.state,
            'session_duration': time.time() - self.last_analysis_time
        }

class DeadliftRepDetector:
    def __init__(self):
        self.state = "STANDING"
        self.rep_count = 0
        self.last_hip_y = None
        
    def update(self, hip_y, torso_angle, hip_angle, knee_angle):
        if self.last_hip_y is None:
            self.last_hip_y = hip_y
            return False, self.state
        
        hip_moving_down = hip_y > self.last_hip_y + 0.01
        hip_moving_up = hip_y < self.last_hip_y - 0.01
        
        if self.state == "STANDING" and torso_angle < 150 and hip_moving_down:
            self.state = "DESCENDING"
        elif self.state == "DESCENDING" and torso_angle < 100:
            self.state = "BOTTOM"
        elif self.state == "BOTTOM" and hip_moving_up:
            self.state = "ASCENDING"
        elif self.state == "ASCENDING" and torso_angle > 150:
            self.state = "STANDING"
            self.rep_count += 1
            self.last_hip_y = hip_y
            return True, self.state
        
        self.last_hip_y = hip_y
        return False, self.state

# Helper functions
def angle_with_vertical(dx, dy):
    try:
        dot = dx * 0 + dy * (-1)
        mag_v = math.hypot(dx, dy)
        if mag_v == 0:
            return 0.0
        cos_a = max(min(dot / mag_v, 1), -1)
        return math.degrees(math.acos(cos_a))
    except:
        return 0.0  # Fixed indentation here

def angle_between_points(a, b, c):
    try:
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cos_angle = max(min(cos_angle, 1), -1)
        return math.degrees(np.arccos(cos_angle))
    except:
        return 0.0

# Global analysis instance
analysis_engine = DeadliftAnalysis()

# HTML Routes
@app.route('/')
def index():
    """Main web interface"""
    stats = analysis_engine.get_stats()
    return render_template('index.html', 
                         total_reps=stats['total_reps'],
                         total_frames=stats['total_frames'],
                         current_state=stats['current_state'])

@app.route('/dashboard')
def dashboard():
    """Advanced dashboard"""
    stats = analysis_engine.get_stats()
    recent_analysis = list(analysis_engine.analysis_history)[-10:]  # Last 10 analyses
    return render_template('dashboard.html',
                         stats=stats,
                         recent_analysis=recent_analysis)

@app.route('/api')
def api_docs():
    """API documentation page"""
    return render_template('api_docs.html')

# API Routes
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'frames_processed': analysis_engine.frame_count,
        'total_reps': analysis_engine.rep_detector.rep_count,
        'current_state': analysis_engine.rep_detector.state
    })

@app.route('/api/reset', methods=['POST'])
def reset_analysis():
    analysis_engine.rep_detector = DeadliftRepDetector()
    analysis_engine.analysis_history.clear()
    return jsonify({
        'message': 'Analysis reset successfully',
        'timestamp': time.time()
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    stats = analysis_engine.get_stats()
    return jsonify(stats)

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    try:
        start_time = time.time()
        
        # Parse JSON data
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({'error': 'No frame data provided'}), 400
        
        # Decode base64 image
        frame_data = data['frame']
        if frame_data.startswith('data:image'):
            frame_data = frame_data.split(',')[1]
        
        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Could not decode image'}), 400
        
        # Process frame
        result = analysis_engine.process_frame(frame)
        
        if result is None:
            return jsonify({'error': 'Frame processing failed'}), 500
        
        # Add processing time
        result['processing_time'] = time.time() - start_time
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in process_frame: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Deadlift Analysis Server...")
    print("📊 Web Interface: http://localhost:5005")
    print("🔗 API Endpoint: http://localhost:5005/api/process_frame")
    print("❤️  Health Check: http://localhost:5005/api/health")
    app.run(host='0.0.0.0', port=5005, debug=True)
