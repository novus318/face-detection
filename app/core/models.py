import numpy as np
from typing import List, Dict, Any
import logging
import cv2

logger = logging.getLogger(__name__)


class ONNXDetector:
    def __init__(self, model_path: str = None, providers: List[str] = None):
        if providers is None:
            providers = ["CPUExecutionProvider"]
        
        try:
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(name="buffalo_l", providers=providers)
            self.app.prepare(ctx_id=0, det_size=(320, 320))
            self.use_insightface = True
            logger.info("Face detector loaded (InsightFace buffalo_l)")
        except Exception as e:
            logger.error(f"Failed to load InsightFace: {e}")
            self.use_insightface = False

    def detect(self, img: np.ndarray, img_size: int = 640, conf_threshold: float = 0.45) -> List[Dict[str, Any]]:
        if not self.use_insightface:
            return []
        
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
            
        faces = self.app.get(img_rgb)
        
        detections = []
        for face in faces:
            conf = float(face.confidence) if face.confidence is not None else 0.5
            if conf < conf_threshold:
                continue
                
            bbox = face.bbox
            landmarks = []
            if hasattr(face, 'kps') and face.kps is not None:
                for kp in face.kps[:5]:
                    landmarks.append([float(kp[0]), float(kp[1])])
            else:
                landmarks = self._estimate_landmarks(bbox[0], bbox[1], bbox[2], bbox[3])
            
            detections.append({
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                "confidence": conf,
                "landmarks": landmarks
            })
        
        return detections

    def _estimate_landmarks(self, x1: float, y1: float, x2: float, y2: float) -> List[List[float]]:
        w = x2 - x1
        h = y2 - y1
        
        landmarks = [
            [x1 + w * 0.30, y1 + h * 0.35],
            [x1 + w * 0.70, y1 + h * 0.35],
            [x1 + w * 0.50, y1 + h * 0.55],
            [x1 + w * 0.35, y1 + h * 0.75],
            [x1 + w * 0.65, y1 + h * 0.75]
        ]
        return landmarks