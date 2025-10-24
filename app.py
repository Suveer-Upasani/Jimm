import cv2
import mediapipe as mp
import numpy as np
import time
import math
import base64
import json
import asyncio
import logging
import os
from flask import Flask, Response, jsonify, request, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from collections import deque
import threading
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaRecorder
from av import VideoFrame
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
CORS(app)

# Store active connections
active_connections = {}

# Pose settings
pose_settings = {
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5,
    'model_complexity': 1
}

def create_pose_instance():
    """Create a fresh MediaPipe Pose instance"""
    return mp_pose.Pose(
        min_detection_confidence=pose_settings['min_detection_confidence'],
        min_tracking_confidence=pose_settings['min_tracking_confidence'],
        model_complexity=pose_settings['model_complexity']
    )

class PoseStabilityFilter:
    """Filter to smooth pose landmarks and reduce jitter"""
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.landmark_history = deque(maxlen=window_size)

    def smooth_landmarks(self, landmarks):
        """Apply smoothing filter to reduce jitter"""
        if not landmarks:
            return landmarks

        self.landmark_history.append(landmarks)

        if len(self.landmark_history) < 2:
            return landmarks

        smoothed = []
        for i in range(len(landmarks)):
            avg_x = sum(frame[i].x for frame in self.landmark_history) / len(self.landmark_history)
            avg_y = sum(frame[i].y for frame in self.landmark_history) / len(self.landmark_history)
            avg_z = sum(frame[i].z for frame in self.landmark_history) / len(self.landmark_history)
            avg_vis = sum(frame[i].visibility for frame in self.landmark_history) / len(self.landmark_history)

            new_lm = type(landmarks[0])()
            new_lm.x = avg_x
            new_lm.y = avg_y
            new_lm.z = avg_z
            new_lm.visibility = avg_vis
            smoothed.append(new_lm)

        return smoothed

class AdvancedBiomechanicalAnalyzer:
    """Enhanced biomechanical analysis with joint angles and forces"""
    def __init__(self):
        self.body_weight = 75
        self.barbell_weight = 60

    def calculate_joint_angles(self, landmarks, width, height):
        """Calculate all relevant joint angles with error checking"""
        try:
            def safe_convert(lm):
                if lm.visibility < 0.5:
                    return None
                return (lm.x * width, lm.y * height)

            shoulder = safe_convert(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
            hip = safe_convert(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
            knee = safe_convert(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
            ankle = safe_convert(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

            if not all([shoulder, hip, knee, ankle]):
                return {}

            back_angle = self.angle_between_points(shoulder, hip, (hip[0], hip[1] - 100))
            hip_angle = self.angle_between_points(shoulder, hip, knee)
            knee_angle = self.angle_between_points(hip, knee, ankle)
            shin_angle = self.angle_between_points(hip, knee, (knee[0], knee[1] + 100))

            return {
                'back_angle': back_angle,
                'hip_angle': hip_angle,
                'knee_angle': knee_angle,
                'shin_angle': shin_angle
            }
        except Exception as e:
            logger.error(f"Angle calculation error: {e}")
            return {}

    def angle_between_points(self, a, b, c):
        """More robust angle calculation"""
        try:
            ba = np.array([a[0]-b[0], a[1]-b[1]])
            bc = np.array([c[0]-b[0], c[1]-b[1]])

            dot_product = np.dot(ba, bc)
            mag_ba = np.linalg.norm(ba)
            mag_bc = np.linalg.norm(bc)

            if mag_ba == 0 or mag_bc == 0:
                return 0.0

            cos_angle = dot_product / (mag_ba * mag_bc)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)

            angle_rad = np.arccos(cos_angle)
            return np.degrees(angle_rad)
        except:
            return 0.0

    def calculate_spinal_loading(self, back_angle, barbell_weight):
        """More accurate spinal loading calculation"""
        try:
            back_angle_rad = np.radians(back_angle)
            shear_force = barbell_weight * np.sin(back_angle_rad)
            compression_force = barbell_weight * np.cos(back_angle_rad)
            body_compression = self.body_weight * 0.6
            total_compression = compression_force + body_compression

            return {
                'total_compression': total_compression,
                'shear_force': shear_force,
                'compression_force': compression_force
            }
        except:
            return {'total_compression': 0, 'shear_force': 0, 'compression_force': 0}

class AdvancedDeadliftRepDetector:
    """Enhanced rep detection with adaptive thresholds"""
    def __init__(self):
        self.state = "STANDING"
        self.rep_count = 0
        self.rep_start_time = None
        self.last_state_change = time.time()
        self.state_persistence_time = 0.3
        self.hip_height_history = deque(maxlen=30)
        self.adaptive_thresholds = {
            'standing_hip_height': 0.3,
            'bottom_hip_height': 0.7
        }

    def update_adaptive_thresholds(self, current_hip_height):
        """Update thresholds based on recent movement"""
        self.hip_height_history.append(current_hip_height)

        if len(self.hip_height_history) > 10:
            min_hip = min(self.hip_height_history)
            max_hip = max(self.hip_height_history)
            range_hip = max_hip - min_hip

            if range_hip > 0.1:
                self.adaptive_thresholds['standing_hip_height'] = min_hip + range_hip * 0.2
                self.adaptive_thresholds['bottom_hip_height'] = min_hip + range_hip * 0.8

    def detect_rep_state(self, landmarks, angles, timestamp):
        """Enhanced rep detection with multiple criteria"""
        if not landmarks or not angles:
            return False, self.state

        hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
        self.update_adaptive_thresholds(hip_y)

        if timestamp - self.last_state_change < self.state_persistence_time:
            return False, self.state

        back_angle = angles.get('back_angle', 0)
        hip_angle = angles.get('hip_angle', 0)
        knee_angle = angles.get('knee_angle', 0)

        new_state = self.state
        rep_completed = False

        if self.state == "STANDING":
            if back_angle > 45 and hip_y > self.adaptive_thresholds['standing_hip_height'] and hip_angle > 150:
                new_state = "DESCENDING"
                self.rep_start_time = timestamp

        elif self.state == "DESCENDING":
            if hip_y >= self.adaptive_thresholds['bottom_hip_height'] and knee_angle < 100 and back_angle > 70:
                new_state = "BOTTOM"

        elif self.state == "BOTTOM":
            if hip_y < self.adaptive_thresholds['bottom_hip_height'] - 0.05:
                new_state = "ASCENDING"

        elif self.state == "ASCENDING":
            if hip_y <= self.adaptive_thresholds['standing_hip_height'] and back_angle < 30 and hip_angle > 160:
                new_state = "STANDING"
                self.rep_count += 1
                rep_completed = True

        if new_state != self.state:
            self.state = new_state
            self.last_state_change = timestamp

        return rep_completed, self.state

class FormQualityAnalyzer:
    """Comprehensive form quality assessment"""
    def __init__(self):
        self.ideal_ranges = {
            'back_angle_start': (15, 30),
            'hip_height_relative': (0.4, 0.6),
            'knee_angle_start': (140, 160),
            'symmetry_threshold': 0.9
        }

    def calculate_comprehensive_quality(self, landmarks, angles, reference_points):
        """Calculate form quality using multiple factors"""
        quality_factors = {}

        back_angle = angles.get('back_angle', 0)
        ideal_back = (self.ideal_ranges['back_angle_start'][0] + self.ideal_ranges['back_angle_start'][1]) / 2
        back_quality = 100 - min(100, abs(back_angle - ideal_back) / ideal_back * 100)
        quality_factors['back_angle'] = max(0, back_quality)

        hip_alignment = self.assess_hip_alignment(landmarks, reference_points)
        quality_factors['hip_position'] = hip_alignment

        symmetry_quality = self.assess_symmetry(landmarks)
        quality_factors['symmetry'] = symmetry_quality

        weights = {
            'back_angle': 0.4,
            'hip_position': 0.35,
            'symmetry': 0.25
        }

        overall_quality = sum(quality_factors[factor] * weights[factor] for factor in quality_factors)

        return {
            'overall': max(0, min(100, overall_quality)),
            'factors': quality_factors
        }

    def assess_hip_alignment(self, landmarks, reference_points):
        """Check hip alignment with reference"""
        try:
            hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            vx = reference_points.get('vertical_line_x', 320)
            hip_x = hip.x * 640

            deviation = abs(hip_x - vx)
            max_deviation = 100
            alignment_quality = 100 - min(100, (deviation / max_deviation) * 100)
            return max(0, alignment_quality)
        except:
            return 50.0

    def assess_symmetry(self, landmarks):
        """Check left-right symmetry"""
        try:
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

            hip_height_diff = abs(left_hip.y - right_hip.y)
            shoulder_height_diff = abs(left_shoulder.y - right_shoulder.y)

            max_diff = 0.05
            hip_symmetry = 100 - min(100, (hip_height_diff / max_diff) * 100)
            shoulder_symmetry = 100 - min(100, (shoulder_height_diff / max_diff) * 100)

            return max(0, (hip_symmetry + shoulder_symmetry) / 2)
        except:
            return 50.0

def calculate_landmark_confidence(landmarks, required_landmarks):
    """Calculate confidence score for pose detection"""
    confidence = 0.0
    visible_count = 0

    for idx in required_landmarks:
        if idx < len(landmarks) and landmarks[idx].visibility > 0.7:
            confidence += landmarks[idx].visibility
            visible_count += 1

    return confidence / len(required_landmarks) if visible_count > 0 else 0.0

def create_mirrored_landmarks(original_landmarks):
    """Create a list of mirrored landmarks by flipping x coordinates"""
    try:
        mirrored_landmarks = []
        for landmark in original_landmarks:
            mirrored_landmark = type(landmark)()
            mirrored_landmark.x = 1.0 - landmark.x
            mirrored_landmark.y = landmark.y
            mirrored_landmark.z = landmark.z
            mirrored_landmark.visibility = landmark.visibility
            mirrored_landmarks.append(mirrored_landmark)
        return mirrored_landmarks
    except:
        return []

def draw_mesh_skeleton(image, landmarks):
    """Draw enhanced mesh-style skeleton visualization"""
    if not landmarks:
        return

    try:
        h, w, _ = image.shape

        # Define pose connections with colors
        connection_colors = {
            'torso': (0, 255, 255),      # Cyan
            'left_arm': (255, 100, 100),  # Light red
            'right_arm': (100, 255, 100), # Light green
            'left_leg': (255, 200, 0),    # Orange
            'right_leg': (200, 100, 255)  # Purple
        }

        # Group connections by body part
        torso_connections = [
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
            (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP)
        ]

        left_arm_connections = [
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
            (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST)
        ]

        right_arm_connections = [
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
            (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST)
        ]

        left_leg_connections = [
            (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
            (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
            (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_FOOT_INDEX)
        ]

        right_leg_connections = [
            (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
            (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
            (mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)
        ]

        # Draw connections with gradient effect
        def draw_connection_group(connections, color, thickness=4):
            for start_lm, end_lm in connections:
                start_idx = start_lm.value
                end_idx = end_lm.value

                if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                    continue

                start = landmarks[start_idx]
                end = landmarks[end_idx]

                if start.visibility < 0.5 or end.visibility < 0.5:
                    continue

                start_px = (int(start.x * w), int(start.y * h))
                end_px = (int(end.x * w), int(end.y * h))

                # Draw thick glowing line
                cv2.line(image, start_px, end_px, color, thickness + 2, cv2.LINE_AA)
                cv2.line(image, start_px, end_px, (255, 255, 255), thickness, cv2.LINE_AA)

        # Draw all connection groups
        draw_connection_group(torso_connections, connection_colors['torso'], 5)
        draw_connection_group(left_arm_connections, connection_colors['left_arm'], 4)
        draw_connection_group(right_arm_connections, connection_colors['right_arm'], 4)
        draw_connection_group(left_leg_connections, connection_colors['left_leg'], 5)
        draw_connection_group(right_leg_connections, connection_colors['right_leg'], 5)

        # Draw joint points with glow effect
        for i, landmark in enumerate(landmarks):
            if landmark.visibility < 0.5:
                continue

            px = (int(landmark.x * w), int(landmark.y * h))

            # Determine joint color based on importance
            if i in [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                     mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value]:
                joint_color = (0, 255, 255)  # Cyan for key joints
                radius = 8
            elif i in [mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.RIGHT_KNEE.value]:
                joint_color = (255, 255, 0)  # Yellow for knees
                radius = 7
            else:
                joint_color = (100, 255, 100)  # Light green for other joints
                radius = 5

            # Glow effect
            cv2.circle(image, px, radius + 3, joint_color, 2, cv2.LINE_AA)
            cv2.circle(image, px, radius, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(image, px, radius - 2, joint_color, -1, cv2.LINE_AA)

    except Exception as e:
        logger.error(f"Error drawing mesh skeleton: {e}")

def calculate_deadlift_reference_points(ankle_px, knee_px, shoulder_px, hip_px, frame_height):
    """Calculate reference points for form analysis"""
    try:
        mid_foot_x = ankle_px[0]
        knee_height = knee_px[1]
        shoulder_height = shoulder_px[1]
        optimal_hip_height = knee_height + (shoulder_height - knee_height) * 0.6

        return {
            'vertical_line_x': mid_foot_x,
            'optimal_hip_height': optimal_hip_height,
            'mid_foot': (mid_foot_x, ankle_px[1])
        }
    except:
        return {
            'vertical_line_x': 320,
            'optimal_hip_height': 240,
            'mid_foot': (320, 400)
        }

def draw_reference_overlay(img, reference_points, hip_px, form_quality):
    """Draw reference lines and form indicators"""
    try:
        vx = reference_points['vertical_line_x']
        hip_y = int(reference_points['optimal_hip_height'])

        # Draw vertical bar path line
        cv2.line(img, (vx, 0), (vx, img.shape[0]), (0, 255, 255), 2, cv2.LINE_AA)

        # Draw horizontal hip reference line
        cv2.line(img, (vx - 80, hip_y), (vx + 80, hip_y), (0, 255, 255), 2, cv2.LINE_AA)

        # Draw reference crosshair
        cv2.circle(img, (vx, hip_y), 60, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(img, (vx, hip_y), 10, (0, 0, 255), -1, cv2.LINE_AA)

        # Draw current hip position indicator
        cv2.circle(img, hip_px, 12, (255, 50, 50), -1, cv2.LINE_AA)
        cv2.circle(img, hip_px, 12, (255, 255, 255), 2, cv2.LINE_AA)

        # Connection lines with transparency effect
        cv2.line(img, hip_px, (vx, hip_px[1]), (255, 100, 100), 2, cv2.LINE_AA)

        # Labels
        cv2.putText(img, "BAR PATH", (vx + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "IDEAL HIP", (vx + 15, hip_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

        # Mid-foot marker
        cv2.circle(img, reference_points['mid_foot'], 8, (0, 255, 0), -1, cv2.LINE_AA)

    except Exception as e:
        logger.error(f"Error drawing reference overlay: {e}")

def draw_trajectories(img, trajectories):
    """Draw motion trajectories for key joints"""
    colors = {
        'hip': (0, 255, 255),
        'shoulder': (255, 255, 0),
        'knee': (255, 0, 255),
        'ankle': (0, 255, 0)
    }

    for landmark_name, points in trajectories.items():
        if len(points) > 1:
            color = colors.get(landmark_name, (255, 255, 255))

            # Draw trajectory path with fade effect
            for i in range(1, len(points)):
                alpha = i / len(points)
                thickness = max(1, int(3 * alpha))
                cv2.line(img, points[i-1], points[i], color, thickness, cv2.LINE_AA)

            # Draw current position
            if points:
                cv2.circle(img, points[-1], 8, color, -1, cv2.LINE_AA)
                cv2.circle(img, points[-1], 10, (255, 255, 255), 2, cv2.LINE_AA)

def draw_hud_overlay(img, analysis_data):
    """Draw heads-up display with metrics"""
    try:
        h, w = img.shape[:2]

        # Semi-transparent background for HUD
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (250, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        # Display metrics
        y_offset = 35
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Rep count (large)
        cv2.putText(img, f"REPS: {analysis_data['reps']}", (20, y_offset),
                    font, 1.0, (0, 255, 0), 3, cv2.LINE_AA)
        y_offset += 35

        # State
        state_color = (255, 255, 0) if analysis_data['state'] != "STANDING" else (0, 255, 0)
        cv2.putText(img, f"State: {analysis_data['state']}", (20, y_offset),
                    font, 0.6, state_color, 2, cv2.LINE_AA)
        y_offset += 30

        # Form quality bar
        form_quality = analysis_data['form_quality']
        if form_quality > 80:
            quality_color = (0, 255, 0)
        elif form_quality > 60:
            quality_color = (0, 255, 255)
        else:
            quality_color = (0, 0, 255)

        cv2.putText(img, f"Form: {form_quality:.0f}%", (20, y_offset),
                    font, 0.6, quality_color, 2, cv2.LINE_AA)

        # Form quality progress bar
        bar_x, bar_y = 20, y_offset + 10
        bar_width = 210
        bar_height = 15
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + int(bar_width * form_quality / 100), bar_y + bar_height),
                     quality_color, -1)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)

        # Confidence indicator (bottom right)
        if 'confidence' in analysis_data:
            conf_text = f"Confidence: {analysis_data['confidence']:.0%}"
            cv2.putText(img, conf_text, (w - 200, h - 20), font, 0.5, (100, 255, 100), 1, cv2.LINE_AA)

    except Exception as e:
        logger.error(f"Error drawing HUD overlay: {e}")

class VideoProcessor:
    """Process video frames and perform pose analysis"""
    def __init__(self, connection_id):
        self.connection_id = connection_id
        self.pose_filter = PoseStabilityFilter()
        self.advanced_analyzer = AdvancedBiomechanicalAnalyzer()
        self.form_analyzer = FormQualityAnalyzer()
        self.advanced_rep_detector = AdvancedDeadliftRepDetector()
        
        self.trajectories = {
            'hip': deque(maxlen=50),
            'shoulder': deque(maxlen=50),
            'knee': deque(maxlen=50),
            'ankle': deque(maxlen=50)
        }
        
        self.pose_instance = create_pose_instance()
        self.is_processing = True

    def process_frame(self, frame):
        """Process a single frame and return analysis results"""
        try:
            # Resize frame for consistent processing
            frame = cv2.resize(frame, (640, 480))
            
            # Create black canvas for mesh visualization
            annotated_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = self.pose_instance.process(rgb_frame)

            analysis_data = {
                'pose_detected': False,
                'reps': self.advanced_rep_detector.rep_count,
                'state': self.advanced_rep_detector.state,
                'form_quality': 0,
                'spinal_load': 0,
                'confidence': 0,
                'timestamp': time.time()
            }

            if pose_results.pose_landmarks:
                mirrored_landmarks = create_mirrored_landmarks(pose_results.pose_landmarks)

                if mirrored_landmarks:
                    smoothed_landmarks = self.pose_filter.smooth_landmarks(mirrored_landmarks)

                    required_landmarks = [
                        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                        mp_pose.PoseLandmark.LEFT_HIP.value,
                        mp_pose.PoseLandmark.LEFT_KNEE.value,
                        mp_pose.PoseLandmark.LEFT_ANKLE.value,
                        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                        mp_pose.PoseLandmark.RIGHT_HIP.value
                    ]

                    confidence = calculate_landmark_confidence(smoothed_landmarks, required_landmarks)
                    analysis_data['confidence'] = confidence

                    if confidence >= 0.6:
                        analysis_data['pose_detected'] = True

                        # Draw mesh skeleton
                        draw_mesh_skeleton(annotated_frame, smoothed_landmarks)

                        # Calculate angles
                        angles = self.advanced_analyzer.calculate_joint_angles(smoothed_landmarks, 640, 480)

                        # Get key point positions
                        def lm_to_pixel(lm):
                            return (int(lm.x * 640), int(lm.y * 480))

                        hip_px = lm_to_pixel(smoothed_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
                        knee_px = lm_to_pixel(smoothed_landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
                        ankle_px = lm_to_pixel(smoothed_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
                        shoulder_px = lm_to_pixel(smoothed_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])

                        # Update trajectories
                        self.trajectories['hip'].append(hip_px)
                        self.trajectories['shoulder'].append(shoulder_px)
                        self.trajectories['knee'].append(knee_px)
                        self.trajectories['ankle'].append(ankle_px)

                        # Calculate reference points
                        reference_points = calculate_deadlift_reference_points(ankle_px, knee_px, shoulder_px, hip_px, 480)

                        # Form quality analysis
                        form_analysis = self.form_analyzer.calculate_comprehensive_quality(smoothed_landmarks, angles, reference_points)
                        analysis_data['form_quality'] = form_analysis['overall']

                        # Rep detection
                        rep_completed, current_state = self.advanced_rep_detector.detect_rep_state(
                            smoothed_landmarks, angles, time.time()
                        )

                        analysis_data['reps'] = self.advanced_rep_detector.rep_count
                        analysis_data['state'] = current_state

                        # Spinal loading
                        if angles.get('back_angle'):
                            spinal_forces = self.advanced_analyzer.calculate_spinal_loading(
                                angles['back_angle'], self.advanced_analyzer.barbell_weight
                            )
                            analysis_data['spinal_load'] = spinal_forces['total_compression']

                        # Draw overlays
                        draw_trajectories(annotated_frame, self.trajectories)
                        draw_reference_overlay(annotated_frame, reference_points, hip_px, form_analysis['overall'])
                        draw_hud_overlay(annotated_frame, analysis_data)

            # Encode annotated frame
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            analysis_data['annotated_frame'] = f"data:image/jpeg;base64,{img_base64}"

            return analysis_data

        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return None

    def stop_processing(self):
        """Clean up resources"""
        self.is_processing = False
        try:
            self.pose_instance.close()
        except:
            pass

# WebRTC Connection Handler
class WebRTCConnection:
    def __init__(self, connection_id):
        self.connection_id = connection_id
        self.pc = RTCPeerConnection()
        self.video_processor = VideoProcessor(connection_id)
        self.setup_connection()

    def setup_connection(self):
        """Set up WebRTC connection with video track"""
        @self.pc.on("track")
        def on_track(track):
            logger.info(f"Track {track.kind} received")
            
            if track.kind == "video":
                # Process video frames
                asyncio.ensure_future(self.handle_video_track(track))

        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Connection state is {self.pc.connectionState}")
            if self.pc.connectionState == "failed" or self.pc.connectionState == "closed":
                await self.cleanup()

    async def handle_video_track(self, track):
        """Handle incoming video track and process frames"""
        logger.info("Starting video frame processing")
        
        while True:
            try:
                frame = await track.recv()
                
                # Convert frame to OpenCV format
                img = frame.to_ndarray(format="bgr24")
                
                # Process frame
                analysis_data = self.video_processor.process_frame(img)
                
                if analysis_data:
                    # Send analysis results via WebSocket
                    await socketio.emit('analysis_update', {
                        'connection_id': self.connection_id,
                        'data': analysis_data
                    })
                    
            except Exception as e:
                logger.error(f"Error processing video frame: {e}")
                break

    async def cleanup(self):
        """Clean up connection resources"""
        self.video_processor.stop_processing()
        await self.pc.close()
        if self.connection_id in active_connections:
            del active_connections[self.connection_id]

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'message': 'Connected to deadlift analysis server'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")
    # Clean up any active connections for this client

@socketio.on('start_analysis')
def handle_start_analysis(data):
    """Start analysis session"""
    connection_id = data.get('connection_id', str(uuid.uuid4()))
    
    # Create new WebRTC connection
    webrtc_conn = WebRTCConnection(connection_id)
    active_connections[connection_id] = webrtc_conn
    
    emit('analysis_started', {
        'connection_id': connection_id,
        'message': 'Analysis session started'
    })

@socketio.on('stop_analysis')
def handle_stop_analysis(data):
    """Stop analysis session"""
    connection_id = data.get('connection_id')
    
    if connection_id in active_connections:
        asyncio.ensure_future(active_connections[connection_id].cleanup())
        emit('analysis_stopped', {
            'connection_id': connection_id,
            'message': 'Analysis session stopped'
        })
    else:
        emit('error', {'message': 'Connection not found'})

@socketio.on('webrtc_offer')
async def handle_webrtc_offer(data):
    """Handle WebRTC offer from client"""
    try:
        connection_id = data.get('connection_id')
        offer = data.get('offer')
        
        if connection_id not in active_connections:
            emit('error', {'message': 'Connection not found'})
            return
        
        webrtc_conn = active_connections[connection_id]
        
        # Set remote description
        await webrtc_conn.pc.setRemoteDescription(
            RTCSessionDescription(sdp=offer['sdp'], type=offer['type'])
        )
        
        # Create answer
        answer = await webrtc_conn.pc.createAnswer()
        await webrtc_conn.pc.setLocalDescription(answer)
        
        # Send answer back to client
        emit('webrtc_answer', {
            'connection_id': connection_id,
            'answer': {
                'sdp': webrtc_conn.pc.localDescription.sdp,
                'type': webrtc_conn.pc.localDescription.type
            }
        })
        
    except Exception as e:
        logger.error(f"Error handling WebRTC offer: {e}")
        emit('error', {'message': f'WebRTC error: {str(e)}'})

@app.route('/')
def index():
    return jsonify({
        'message': 'Advanced Deadlift Analysis API with WebRTC',
        'status': 'ready',
        'version': '3.0',
        'features': [
            'WebRTC real-time video streaming',
            'Mesh skeleton visualization',
            'Adaptive rep detection',
            'Multi-factor form analysis',
            'Real-time biomechanics',
            'Motion trajectories'
        ],
        'endpoints': {
            'WebSocket /': 'Real-time communication',
            'GET /health': 'Health check'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'active_connections': len(active_connections),
        'service': 'deadlift-analysis-webrtc-api',
        'version': '3.0'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5005))
    print("=" * 60)
    print("🏋️  ADVANCED DEADLIFT ANALYSIS API v3.0 (WebRTC)")
    print("=" * 60)
    print(f"🚀 Server starting on port {port}")
    print(f"📡 WebSocket ready for real-time communication")
    print(f"🎨 Features: WebRTC streaming, mesh visualization, real-time analysis")
    print("=" * 60)
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
