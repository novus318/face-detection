from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class FacePosition(BaseModel):
    roll: float
    yaw: float
    pitch: float
    face_center_x: float
    face_center_y: float
    face_size_ratio: float
    is_frontal: bool


class FaceMetrics(BaseModel):
    blurriness: float
    is_blurry: bool
    position: FacePosition


class M2ImageResult(BaseModel):
    index: int
    score: float
    face_detected: bool
    confidence: Optional[float] = None
    metrics: Optional[FaceMetrics] = None


class VerifyResponse(BaseModel):
    match_score: float = Field(..., description="Best cosine similarity score")
    match_percentage: float = Field(..., description="Best match score as percentage")
    is_match: bool = Field(..., description="Whether faces match")
    m1_face_detected: bool = Field(..., description="Face detected in m1")
    m2_best_frame_index: int = Field(..., description="Best matching frame index from m2")
    m2_all_scores: Optional[List[M2ImageResult]] = Field(None, description="All m2 image scores and details")
    processing_ms: int = Field(..., description="Total processing time in milliseconds")
    m1_face_metrics: Optional[FaceMetrics] = Field(None, description="Face quality metrics for m1")
    m2_face_metrics: Optional[FaceMetrics] = Field(None, description="Face quality metrics for best m2 match")