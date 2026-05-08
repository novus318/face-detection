from pydantic import BaseModel, Field
from typing import Optional


class VerifyRequest(BaseModel):
    m1_url: str = Field(..., description="URL of the first image (e.g., passport)")
    m2_urls: list[str] = Field(..., description="URLs of the second images (e.g., selfie frames)")
    options: Optional[dict] = Field(
        default=None,
        description="Optional parameters: return_images, match_threshold"
    )


class DocIntelRequest(BaseModel):
    document_url: str = Field(..., description="URL of the document image (passport, ID card, license)")