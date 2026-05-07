from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
from app.config import get_settings
from app.api.v1.router import api_router
from app.api.v1.verify import set_detector, set_embedding_model
from app.core.models import ONNXDetector
from app.core.embedding import MobileFaceNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Loading models...")
    
    try:
        detector = ONNXDetector(
            model_path=settings.model_path,
            providers=["CPUExecutionProvider"]
        )
        set_detector(detector)
        logger.info("YOLOv5n-face detector loaded")

        embedding_model = MobileFaceNet()
        set_embedding_model(embedding_model)
        logger.info("InsightFace embedding model loaded")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    logger.info("Shutting down...")


app = FastAPI(
    title="Face Verification API",
    description="Fast face verification API optimized for Railway.com",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "face-verification"}


@app.get("/")
async def root():
    return {"message": "Face Verification API", "version": "1.0.0"}