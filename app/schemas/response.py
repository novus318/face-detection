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


class DocIntelResponse(BaseModel):
    document_type: str = Field(..., description="Document type: passport, driversLicense, nationalIdCard")
    country: str = Field(..., description="Country/region of issuance")
    country_code: Optional[str] = Field(None, description="ISO country code")
    first_name: Optional[str] = Field(None, description="First name")
    last_name: Optional[str] = Field(None, description="Last name")
    date_of_birth: Optional[str] = Field(None, description="Date of birth (raw)")
    dob_formatted: Optional[str] = Field(None, description="Date of birth (formatted)")
    document_number: Optional[str] = Field(None, description="Document number")
    nationality: Optional[str] = Field(None, description="Nationality")
    sex: Optional[str] = Field(None, description="Sex/M gender")
    expiration_date: Optional[str] = Field(None, description="Expiration date")
    issue_date: Optional[str] = Field(None, description="Issue date")
    region: Optional[str] = Field(None, description="Region/state")
    address: Optional[str] = Field(None, description="Address")
    raw_confidence: float = Field(..., description="Overall confidence score")