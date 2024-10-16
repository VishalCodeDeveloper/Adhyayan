import cv2
import numpy as np
import language_tool_python
from textblob import TextBlob
from fer import FER
import mediapipe as mp

# Initializing Mediapipe components for use
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
hands = mp_hands.Hands(max_num_hands = 2, min_detection_confidence=0.5)

def analyze_video(video_path):
    video_capture = cv2.VideoCapture(video_path)
    emotion_detector = FER()
    emotion_changes_with_time = {}
    eye_contact_data = []

    gesture_counts_per_second = {}
    # hand_gestures = []
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    frame_index = 0

    # Initializing Mediapipe for face, hands, and mesh detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # To store gestures with their timestamps
    detected_gestures = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Converting image into RGB and processing it with Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_detection.process(rgb_frame)
        face_landmarks = face_mesh.process(rgb_frame)
        hand_results = hands.process(rgb_frame)

        # Per Second of video processing
        if frame_index % int(frame_rate) == 0:
            eye_contact_frames = 0
            total_frames_in_second = 0

        if face_results.detections:
            total_frames_in_second += 1

            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                # box around face
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

                # Check face mesh landmark
                if face_landmarks.multi_face_landmarks:
                    left_eye = face_landmarks.multi_face_landmarks[0].landmark[33]
                    right_eye = face_landmarks.multi_face_landmarks[0].landmark[133]
                    nose = face_landmarks.multi_face_landmarks[0].landmark[1]
                    left_eye_coords = (int(left_eye.x * w), int(left_eye.y * h))
                    right_eye_coords = (int(right_eye.x * w), int(right_eye.y * h))
                    nose_coords = (int(nose.x * w), int(nose.y * h))

                    eye_distance = np.linalg.norm(np.array(left_eye_coords) - np.array(right_eye_coords))
                    nose_to_left_eye_distance = np.linalg.norm(np.array(nose_coords) - np.array(left_eye_coords))
                    nose_to_right_eye_distance = np.linalg.norm(np.array(nose_coords) - np.array(right_eye_coords))
                    eye_contact_ratio = (nose_to_left_eye_distance + nose_to_right_eye_distance) / (eye_distance + 1e-5)

                    is_eye_contact = eye_contact_ratio < 0.5
                    eye_contact_frames += is_eye_contact

        # Process hand gestures
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                hand_gesture = detect_hand_gesture(hand_landmarks)
                if hand_gesture != "Unknown Gesture":
                    current_second = frame_index // int(frame_rate)
                    if current_second not in gesture_counts_per_second:
                        gesture_counts_per_second[current_second] = {}
                    if hand_gesture not in gesture_counts_per_second[current_second]:
                        gesture_counts_per_second[current_second][hand_gesture] = 0
                    gesture_counts_per_second[current_second][hand_gesture] += 1

        # Emotion detection data
        emotions = emotion_detector.detect_emotions(frame)
        if emotions:
            dominant_emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
            emotion_changes_with_time[int(frame_index / frame_rate)] = dominant_emotion

        # Eye contact detection data
        if total_frames_in_second > 0 and frame_index % int(frame_rate) == 0 and frame_index > 0:
            eye_contact_detected = eye_contact_frames / total_frames_in_second > 0.5
            eye_contact_data.append((frame_index // int(frame_rate), eye_contact_detected))

        frame_index += 1

    video_capture.release()

    return {
        "emotion_changes": emotion_changes_with_time,  # Assuming this is a list or dict
        "eye_contact": eye_contact_data,                # Assuming this is a list or dict
        "hand_gestures": gesture_counts_per_second      # Assuming this is a list or dict
    }


def detect_hand_gesture(hand_landmarks):
    # landmark coordinates for detecting finger movement
    landmarks = hand_landmarks.landmark
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

    # Calculate distances for gesture recognition
    thumb_index_distance = np.linalg.norm([thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y])
    thumb_pinky_distance = np.linalg.norm([thumb_tip.x - pinky_tip.x, thumb_tip.y - pinky_tip.y])
    
    if thumb_index_distance < 0.1 and thumb_pinky_distance < 0.1:
        return "Open Hand"
    elif thumb_index_distance > 0.15 and thumb_pinky_distance > 0.15:
        return "Closed Fist"
    elif thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y:
        return "Thumbs Up"
    elif index_tip.y < middle_tip.y and index_tip.y < ring_tip.y and middle_tip.y < ring_tip.y:
        return "Peace Sign"
    elif thumb_tip.y < index_tip.y and index_tip.y < middle_tip.y:
        return "Okay Sign"
    elif pinky_tip.y < index_tip.y and pinky_tip.y < ring_tip.y:
        return "Small fingure"
    
    return "Unknown Gesture"