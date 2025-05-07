#!/usr/bin/env python3
"""
Study Buddy - Attention Monitoring System

This application uses computer vision to detect signs of drowsiness or inattention
by tracking eye closure and yawning. It provides real-time alerts to help users
stay focused during study sessions.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import datetime
import argparse
import pygame
import os
from dataclasses import dataclass
from typing import List, Tuple

# Initialize pygame for audio alerts
pygame.mixer.init()


@dataclass
class AlertEvent:
    """Class for tracking alert events"""
    timestamp: datetime.datetime
    event_type: str  # 'eye_closure' or 'yawn'
    duration: float  # Duration in seconds


class StudyBuddy:
    """
    Study Buddy application that uses computer vision to monitor user's attention.
    """

    def __init__(self, eye_closure_threshold: float = 2.0, 
                 yawn_threshold: float = 0.5,
                 log_file: str = "study_buddy_log.txt",
                 alert_sound: str = None):
        """
        Initialize StudyBuddy with detection parameters and resources.
        
        Args:
            eye_closure_threshold: Time in seconds before triggering an eye closure alert
            yawn_threshold: Ratio threshold for yawn detection
            log_file: File path to save alert logs
            alert_sound: Path to sound file for alerts (optional)
        """
        # Detection thresholds
        self.eye_closure_threshold = eye_closure_threshold
        self.yawn_threshold = yawn_threshold
        
        # Alert tracking
        self.eyes_closed_start = None
        self.yawn_start = None
        self.alert_events = []
        self.log_file = log_file
        
        # MediaPipe initialization
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmark indices
        # Eye landmarks
        self.left_eye = [362, 385, 387, 263, 373, 380]  # Left eye landmarks
        self.right_eye = [33, 160, 158, 133, 153, 144]  # Right eye landmarks
        
        # Mouth landmarks for yawn detection
        self.upper_lip = [13, 14, 0, 17]    # Upper lip landmarks
        self.lower_lip = [14, 178, 17, 13]  # Lower lip landmarks
        
        # Alert sound
        self.alert_sound = alert_sound
        if self.alert_sound and os.path.exists(self.alert_sound):
            self.sound = pygame.mixer.Sound(self.alert_sound)
        else:
            # Generate a beep sound if no file provided
            self._generate_beep_sound()
        
        # Font for OpenCV text
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def _generate_beep_sound(self):
        """Generate a simple beep sound for alerts"""
        pygame.mixer.quit()
        pygame.mixer.init(frequency=44100, size=-16, channels=1)
        
        # Create a simple 1s beep
        duration = 1000  # milliseconds
        frequency = 1000  # Hz
        sample_rate = 44100
        bits = -16
        
        # Generate a numpy array of sound samples
        t = np.linspace(0, duration / 1000, int(duration * sample_rate / 1000), False)
        samples = 0.5 * np.sin(2 * np.pi * frequency * t)
        samples = (samples * 32767).astype(np.int16)
        
        # Create sound object
        self.sound = pygame.mixer.Sound(samples)
    
    def calculate_ear(self, landmarks, eye_indices) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for eye openness detection.
        
        Args:
            landmarks: Detected facial landmarks
            eye_indices: Indices of eye landmarks
            
        Returns:
            float: Eye aspect ratio value
        """
        # Get coordinates of eye landmarks
        points = []
        for idx in eye_indices:
            lm = landmarks[idx]
            points.append([lm.x, lm.y])
        points = np.array(points)
        
        # Calculate the vertical distances
        v1 = np.linalg.norm(points[1] - points[5])
        v2 = np.linalg.norm(points[2] - points[4])
        
        # Calculate the horizontal distance
        h = np.linalg.norm(points[0] - points[3])
        
        # Calculate EAR
        ear = (v1 + v2) / (2.0 * h + 1e-6)  # Adding small value to prevent division by zero
        
        return ear
    
    def calculate_mar(self, landmarks) -> float:
        """
        Calculate Mouth Aspect Ratio (MAR) for yawn detection.
        
        Args:
            landmarks: Detected facial landmarks
            
        Returns:
            float: Mouth aspect ratio value
        """
        # Get vertical distance between upper and lower lip
        upper_points = [landmarks[i] for i in self.upper_lip]
        lower_points = [landmarks[i] for i in self.lower_lip]
        
        # Calculate center points of upper and lower lip
        upper_center = np.mean([[p.x, p.y] for p in upper_points], axis=0)
        lower_center = np.mean([[p.x, p.y] for p in lower_points], axis=0)
        
        # Calculate vertical distance
        vertical_dist = np.linalg.norm(upper_center - lower_center)
        
        # Calculate width of mouth - fixing the subtraction of lists
        point1 = np.array([landmarks[self.upper_lip[0]].x, landmarks[self.upper_lip[0]].y])
        point2 = np.array([landmarks[self.upper_lip[2]].x, landmarks[self.upper_lip[2]].y])
        mouth_width = np.linalg.norm(point1 - point2)
        
        # Calculate MAR
        mar = vertical_dist / (mouth_width + 1e-6)
        
        return mar
    
    def log_event(self, event_type: str, duration: float):
        """
        Log alert events to file and memory.
        
        Args:
            event_type: Type of alert ('eye_closure' or 'yawn')
            duration: Duration of the event in seconds
        """
        timestamp = datetime.datetime.now()
        event = AlertEvent(timestamp=timestamp, event_type=event_type, duration=duration)
        self.alert_events.append(event)
        
        # Log to file
        with open(self.log_file, 'a') as f:
            f.write(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')}, {event_type}, {duration:.2f}s\n")
    
    def play_alert(self):
        """Play alert sound"""
        pygame.mixer.Sound.play(self.sound)
    
    def process_frame(self, frame):
        """
        Process a single frame to detect drowsiness and yawning.
        
        Args:
            frame: Image frame from webcam
            
        Returns:
            Processed frame with annotations
        """
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.face_mesh.process(frame_rgb)
        
        # Get frame dimensions
        h, w, _ = frame.shape
        
        # Initialize tracking variables for this frame
        eyes_currently_closed = False
        currently_yawning = False
        
        # Check if face landmarks were detected
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Calculate EAR for both eyes
            left_ear = self.calculate_ear(landmarks, self.left_eye)
            right_ear = self.calculate_ear(landmarks, self.right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Calculate MAR for mouth
            mar = self.calculate_mar(landmarks)
            
            # Remove the facial landmark visualization
            # We'll just detect without drawing the landmarks
            
            # Add EAR and MAR values to frame
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30), self.font, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60), self.font, 0.7, (0, 255, 0), 2)
            
            # Detect closed eyes
            if avg_ear < 0.2:  # Threshold for closed eyes
                eyes_currently_closed = True
                
                # Start timer if eyes just closed
                if self.eyes_closed_start is None:
                    self.eyes_closed_start = time.time()
                else:
                    # Check duration of eye closure
                    duration = time.time() - self.eyes_closed_start
                    
                    # Add visual indicator for closed eyes duration
                    cv2.putText(frame, f"Eyes Closed: {duration:.1f}s", 
                                (10, 90), self.font, 0.7, (0, 0, 255), 2)
                    
                    # Trigger alert if eyes closed for too long
                    if duration >= self.eye_closure_threshold:
                        # Add alert to frame
                        cv2.putText(frame, "ALERT: Wake Up!", 
                                    (int(w/2) - 100, int(h/2)), 
                                    self.font, 1.2, (0, 0, 255), 3)
                        
                        # Play sound alert (only once per event)
                        if duration - self.eye_closure_threshold < 0.1:
                            self.play_alert()
                            self.log_event("eye_closure", duration)
            else:
                # Reset eye closure timer
                self.eyes_closed_start = None
            
            # Detect yawning
            if mar > self.yawn_threshold:
                currently_yawning = True
                
                # Start timer if just started yawning
                if self.yawn_start is None:
                    self.yawn_start = time.time()
                else:
                    # Check duration of yawn
                    duration = time.time() - self.yawn_start
                    
                    # Add visual indicator for yawn duration
                    cv2.putText(frame, f"Yawning: {duration:.1f}s", 
                                (10, 120), self.font, 0.7, (0, 0, 255), 2)
                    
                    # Trigger alert for extended yawning (over 2 seconds)
                    if duration >= 2.0:
                        # Add alert to frame
                        cv2.putText(frame, "ALERT: You're Getting Tired!", 
                                    (int(w/2) - 180, int(h/2) + 40), 
                                    self.font, 1.2, (0, 0, 255), 3)
                        
                        # Play sound alert (only once per event)
                        if duration - 2.0 < 0.1:
                            self.play_alert()
                            self.log_event("yawn", duration)
            else:
                # Reset yawn timer
                self.yawn_start = None
        else:
            # No face detected
            cv2.putText(frame, "No Face Detected", (10, 30), self.font, 0.7, (0, 0, 255), 2)
            
            # Reset timers if face lost
            self.eyes_closed_start = None
            self.yawn_start = None
        
        # Add application info
        cv2.putText(frame, "Study Buddy - Press 'q' to quit", 
                    (10, h - 20), self.font, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def run(self, camera_id: int = 0):
        """
        Main loop to capture and process webcam frames.
        
        Args:
            camera_id: ID of the camera to use
        """
        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write("Timestamp, Event Type, Duration (s)\n")
        
        # Initialize webcam
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        # Set window properties
        cv2.namedWindow("Study Buddy", cv2.WINDOW_NORMAL)
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display the processed frame
                cv2.imshow("Study Buddy", processed_frame)
                
                # Check for exit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            # Release resources
            cap.release()
            cv2.destroyAllWindows()
            self.face_mesh.close()
            print(f"Session ended. Alert events logged to {self.log_file}")
            
            # Print summary
            print(f"Total alerts: {len(self.alert_events)}")
            eye_closure_events = sum(1 for e in self.alert_events if e.event_type == "eye_closure")
            yawn_events = sum(1 for e in self.alert_events if e.event_type == "yawn")
            print(f"Eye closure alerts: {eye_closure_events}")
            print(f"Yawn alerts: {yawn_events}")


def main():
    """Parse command line arguments and run Study Buddy"""
    parser = argparse.ArgumentParser(description="Study Buddy - Attention Monitoring System")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID (default: 0)")
    parser.add_argument("--eye-threshold", type=float, default=2.0, 
                        help="Eye closure threshold in seconds (default: 2.0)")
    parser.add_argument("--yawn-threshold", type=float, default=0.5, 
                        help="Yawn detection threshold (ratio, default: 0.5)")
    parser.add_argument("--log", type=str, default="study_buddy_log.txt", 
                        help="Log file path (default: study_buddy_log.txt)")
    parser.add_argument("--sound", type=str, default=None, 
                        help="Path to custom alert sound file (.wav)")
    
    args = parser.parse_args()
    
    print("Starting Study Buddy...")
    print(f"Eye closure threshold: {args.eye_threshold}s")
    print(f"Yawn threshold: {args.yawn_threshold}")
    print(f"Log file: {args.log}")
    print("Press 'q' to quit")
    
    # Create and run Study Buddy
    buddy = StudyBuddy(
        eye_closure_threshold=args.eye_threshold,
        yawn_threshold=args.yawn_threshold,
        log_file=args.log,
        alert_sound=args.sound
    )
    
    buddy.run(camera_id=args.camera)


if __name__ == "__main__":
    main()