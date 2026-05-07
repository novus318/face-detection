import numpy as np
import cv2
from typing import Dict, Any


def compute_blurriness(img: np.ndarray) -> float:
    if img is None or img.size == 0:
        return 0.0

    try:
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(laplacian_var)
    except Exception:
        return 0.0


def compute_face_position(landmarks: list, img_shape: tuple) -> Dict[str, Any]:
    if not landmarks or len(landmarks) != 5:
        return {
            "roll": 0.0,
            "yaw": 0.0,
            "pitch": 0.0,
            "face_center_x": 0.5,
            "face_center_y": 0.5,
            "face_size_ratio": 0.0,
            "is_frontal": False
        }

    h, w = img_shape[:2]

    nose = landmarks[2]
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    left_mouth = landmarks[3]
    right_mouth = landmarks[4]

    eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    mouth_center = ((left_mouth[0] + right_mouth[0]) / 2, (left_mouth[1] + right_mouth[1]) / 2)

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    roll = np.degrees(np.arctan2(dy, dx))

    face_width = abs(right_eye[0] - left_eye[0]) * 2.5
    face_height = abs(mouth_center[1] - eye_center[1]) * 2.5

    nose_to_eye_ratio = (nose[1] - eye_center[1]) / (face_height / 2) if face_height > 0 else 0

    left_dist = nose[0] - left_eye[0]
    right_dist = right_eye[0] - nose[0]
    asymmetry = abs(left_dist - right_dist) / (face_width / 2) if face_width > 0 else 0

    yaw = np.degrees(np.arcsin(np.clip(asymmetry * 0.5, -1, 1)))

    pitch = (nose_to_eye_ratio - 0.5) * 30

    face_center_x = nose[0] / w if w > 0 else 0.5
    face_center_y = nose[1] / h if h > 0 else 0.5
    face_size_ratio = (face_width * face_height) / (w * h) if w > 0 and h > 0 else 0

    is_frontal = (
        abs(roll) < 15 and
        abs(yaw) < 20 and
        abs(pitch) < 15 and
        asymmetry < 0.2
    )

    return {
        "roll": float(roll),
        "yaw": float(yaw),
        "pitch": float(pitch),
        "face_center_x": float(face_center_x),
        "face_center_y": float(face_center_y),
        "face_size_ratio": float(face_size_ratio),
        "is_frontal": bool(is_frontal)
    }


def compute_face_metrics(img: np.ndarray, landmarks: list, blur_threshold: float = 100.0, orig_img_shape: tuple = None) -> Dict[str, Any]:
    blurriness = compute_blurriness(img)
    
    shape_for_position = orig_img_shape if orig_img_shape else img.shape
    position = compute_face_position(landmarks, shape_for_position)

    return {
        "blurriness": blurriness,
        "is_blurry": blurriness < blur_threshold,
        "position": position
    }