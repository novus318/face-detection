import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    model_path: str = os.path.expanduser("~/.insightface/models/buffalo_l/det_10g.onnx")
    img_size: int = 640
    confidence_threshold: float = 0.45
    max_image_size: int = 1024
    download_timeout: int = 5
    blur_threshold: float = 100.0

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()