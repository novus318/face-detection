from fastapi import APIRouter
from app.api.v1 import verify, doc_intel

api_router = APIRouter()
api_router.include_router(verify.router, tags=["verify"])
api_router.include_router(doc_intel.router, tags=["doc-intel"])