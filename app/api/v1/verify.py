from fastapi import APIRouter, HTTPException, Depends
from app.schemas.request import VerifyRequest
from app.schemas.response import VerifyResponse, FaceMetrics, M2ImageResult
from app.core.models import ONNXDetector
from app.core.embedding import MobileFaceNet
from app.core.downloader import download_multiple
from app.core.alignment import align_face
from app.core.matcher import match_faces, cosine_similarity
from app.core.metrics import compute_face_metrics
from app.config import get_settings
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)
router = APIRouter()

_detector: ONNXDetector = None
_embedding_model: MobileFaceNet = None


def get_detector() -> ONNXDetector:
    global _detector
    if _detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _detector


def get_embedding_model() -> MobileFaceNet:
    global _embedding_model
    if _embedding_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")
    return _embedding_model


def set_detector(detector: ONNXDetector):
    global _detector
    _detector = detector


def set_embedding_model(embedding_model: MobileFaceNet):
    global _embedding_model
    _embedding_model = embedding_model


@router.post("/verify", response_model=VerifyResponse)
async def verify_faces(
    request: VerifyRequest,
    detector: ONNXDetector = Depends(get_detector),
    embedding_model: MobileFaceNet = Depends(get_embedding_model)
):
    settings = get_settings()
    start_time = time.time()
    
    m1_imgs = await download_multiple([request.m1_url], timeout=settings.download_timeout, max_size=settings.max_image_size)
    
    if m1_imgs[0] is None:
        raise HTTPException(status_code=400, detail="Failed to download m1 image")
    
    m1_detections = detector.detect(m1_imgs[0], img_size=settings.img_size, conf_threshold=settings.confidence_threshold)
    
    if not m1_detections:
        raise HTTPException(status_code=400, detail="No face detected in m1 image")
    
    best_m1 = max(m1_detections, key=lambda x: x["confidence"])
    aligned_m1 = align_face(m1_imgs[0], best_m1["landmarks"])
    embedding_m1 = embedding_model.get_embedding(aligned_m1)
    m1_metrics = compute_face_metrics(aligned_m1, best_m1["landmarks"], settings.blur_threshold, m1_imgs[0].shape)
    
    m2_imgs = await download_multiple(request.m2_urls, timeout=settings.download_timeout, max_size=settings.max_image_size)
    
    m2_embeddings = []
    m2_metrics_list = []
    for img in m2_imgs:
        if img is None:
            continue
        
        detections = detector.detect(img, img_size=settings.img_size, conf_threshold=settings.confidence_threshold)
        
        if not detections:
            continue
        
        best = max(detections, key=lambda x: x["confidence"])
        aligned = align_face(img, best["landmarks"])
        embedding = embedding_model.get_embedding(aligned)
        metrics = compute_face_metrics(aligned, best["landmarks"], settings.blur_threshold, img.shape)
        
        m2_embeddings.append(embedding)
        m2_metrics_list.append((best, metrics))
    
    if not m2_embeddings:
        raise HTTPException(status_code=400, detail="No face detected in any m2 image")
    
    match_score, best_idx = match_faces(embedding_m1, m2_embeddings)
    
    threshold = request.options.get("match_threshold", 0.65) if request.options else 0.65
    is_match = match_score >= threshold
    match_percentage = round(match_score * 100, 1)
    
    processing_ms = int((time.time() - start_time) * 1000)
    
    m1_face_metrics = FaceMetrics(
        blurriness=m1_metrics["blurriness"],
        is_blurry=m1_metrics["is_blurry"],
        position=m1_metrics["position"]
    )
    
    best_m2_metrics = m2_metrics_list[best_idx][1] if best_idx >= 0 and best_idx < len(m2_metrics_list) else None
    m2_face_metrics = None
    if best_m2_metrics:
        m2_face_metrics = FaceMetrics(
            blurriness=best_m2_metrics["blurriness"],
            is_blurry=best_m2_metrics["is_blurry"],
            position=best_m2_metrics["position"]
        )
    
    m2_all_scores = []
    for idx, (emb, (det, metrics)) in enumerate(zip(m2_embeddings, m2_metrics_list)):
        score = cosine_similarity(embedding_m1, emb)
        m2_all_scores.append(M2ImageResult(
            index=idx,
            score=round(score, 4),
            face_detected=True,
            confidence=det["confidence"],
            metrics=FaceMetrics(
                blurriness=metrics["blurriness"],
                is_blurry=metrics["is_blurry"],
                position=metrics["position"]
            )
        ))
    
    return VerifyResponse(
        match_score=round(match_score, 4),
        match_percentage=match_percentage,
        is_match=is_match,
        m1_face_detected=True,
        m2_best_frame_index=best_idx,
        m2_all_scores=m2_all_scores,
        processing_ms=processing_ms,
        m1_face_metrics=m1_face_metrics,
        m2_face_metrics=m2_face_metrics,
    )