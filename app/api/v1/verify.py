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
import asyncio
from concurrent.futures import ThreadPoolExecutor

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


def _process_m1(img, detector, embedding_model, img_size, conf_threshold, blur_threshold):
    detections = detector.detect(img, img_size=img_size, conf_threshold=conf_threshold)
    if not detections:
        return None
    
    best = max(detections, key=lambda x: x["confidence"])
    aligned = align_face(img, best["landmarks"])
    embedding = embedding_model.get_embedding(aligned)
    metrics = compute_face_metrics(aligned, best["landmarks"], blur_threshold, img.shape)
    return (embedding, best, metrics)


def _process_single_m2(img, detector, embedding_model, img_size, conf_threshold, blur_threshold):
    if img is None:
        return None
    
    detections = detector.detect(img, img_size=img_size, conf_threshold=conf_threshold)
    
    if not detections:
        return None
    
    best = max(detections, key=lambda x: x["confidence"])
    aligned = align_face(img, best["landmarks"])
    embedding = embedding_model.get_embedding(aligned)
    metrics = compute_face_metrics(aligned, best["landmarks"], blur_threshold, img.shape)
    
    return (embedding, best, metrics)


async def _process_m2_images_concurrent(m2_imgs, detector, embedding_model, settings):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            loop.run_in_executor(
                executor,
                _process_single_m2,
                img, detector, embedding_model,
                settings.img_size, settings.confidence_threshold, settings.blur_threshold
            )
            for img in m2_imgs
        ]
        results = await asyncio.gather(*futures)
    
    m2_embeddings = []
    m2_metrics_list = []
    for result in results:
        if result is not None:
            embedding, det, metrics = result
            m2_embeddings.append(embedding)
            m2_metrics_list.append((det, metrics))
    
    return m2_embeddings, m2_metrics_list


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
    
    m1_result = await asyncio.get_event_loop().run_in_executor(
        None, _process_m1, m1_imgs[0], detector, embedding_model,
        settings.img_size, settings.confidence_threshold, settings.blur_threshold
    )
    
    if m1_result is None:
        raise HTTPException(status_code=400, detail="No face detected in m1 image")
    
    embedding_m1, best_m1, m1_metrics = m1_result
    
    m2_imgs = await download_multiple(request.m2_urls, timeout=settings.download_timeout, max_size=settings.max_image_size)
    
    m2_embeddings, m2_metrics_list = await _process_m2_images_concurrent(
        m2_imgs, detector, embedding_model, settings
    )
    
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