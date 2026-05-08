import aiohttp
import numpy as np
from PIL import Image
import io
import cv2
from typing import Optional
import logging

logger = logging.getLogger(__name__)


async def download_image(url: str, timeout: int = 5, max_size: int = 1024) -> Optional[np.ndarray]:
    try:
        ssl_context = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=ssl_context) as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                if response.status != 200:
                    logger.warning(f"Failed to download {url}: status {response.status}")
                    return None
                
                content = await response.read()
                
                img = Image.open(io.BytesIO(content))
                
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    img = img.resize(new_size, Image.LANCZOS)
                
                img_array = np.array(img)
                
                return img_array
                
    except Exception as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        return None


async def download_multiple(urls: list[str], timeout: int = 5, max_size: int = 1024) -> list[Optional[np.ndarray]]:
    import asyncio
    tasks = [download_image(url, timeout, max_size) for url in urls]
    return await asyncio.gather(*tasks)


async def download_single_image(url: str, timeout: int = 5, max_size: int = 1024) -> Optional[bytes]:
    try:
        ssl_context = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=ssl_context) as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                if response.status != 200:
                    logger.warning(f"Failed to download {url}: status {response.status}")
                    return None
                
                content = await response.read()
                return content
                
    except Exception as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        return None