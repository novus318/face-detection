import numpy as np
import cv2
import logging
import os
import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

logger = logging.getLogger(__name__)


class FaceEmbedding:
    def __init__(self, model_path: str = None):
        self.input_size = (112, 112)
        self.use_insightface = False
        self._app = None
        self._recognition = None
        
        try:
            from insightface.app import FaceAnalysis
            from insightface.model_zoo import model_zoo
            
            self._app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
            self._app.prepare(ctx_id=0, det_size=(640, 640))
            
            model_dir = os.path.expanduser("~/.insightface/models/buffalo_l")
            rec_path = os.path.join(model_dir, "w600k_r50.onnx")
            
            import onnxruntime as ort
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4
            sess_options.inter_op_num_threads = 4
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self._recognition = ort.InferenceSession(rec_path, sess_options, providers=['CPUExecutionProvider'])
            self.input_name = self._recognition.get_inputs()[0].name
            self.output_name = self._recognition.get_outputs()[0].name
            self.use_insightface = True
            logger.info("InsightFace recognition model loaded (w600k_r50)")
        except Exception as e:
            logger.warning(f"Failed to load InsightFace: {e}")

    def get_embedding(self, img: np.ndarray) -> np.ndarray:
        if img is None or img.size == 0:
            return self._empty_embedding()

        try:
            if self.use_insightface:
                return self._get_insightface_embedding(img)
            else:
                return self._simple_embedding(img)
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return self._empty_embedding()

    def _empty_embedding(self):
        return np.zeros(512, dtype=np.float32)

    def _get_insightface_embedding(self, img: np.ndarray) -> np.ndarray:
        if img.shape[:2] != self.input_size:
            img = cv2.resize(img, self.input_size)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)

        try:
            output = self._recognition.run([self.output_name], {self.input_name: img})
            embedding = output[0][0]
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.warning(f"ONNX inference failed: {e}, using fallback")
            return self._simple_embedding(img)

    def _simple_embedding(self, img: np.ndarray) -> np.ndarray:
        if img.shape[:2] != self.input_size:
            img = cv2.resize(img, self.input_size)

        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        hist = hist.flatten()
        if hist.sum() > 0:
            hist = hist / hist.sum()
        
        features = list(hist)
        
        for bh in range(0, 112, 16):
            for bw in range(0, 112, 16):
                block = gray[bh:bh+16, bw:bw+16]
                if block.size > 0:
                    features.append(float(block.mean()) / 255.0)
                    features.append(float(block.std()) / 255.0)

        embedding = np.array(features[:512], dtype=np.float32)
        while len(embedding) < 512:
            embedding = np.concatenate([embedding, embedding[:min(256, len(embedding))]])
        
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding


class MobileFaceNet:
    def __init__(self, model_path: str = None, providers: list = None):
        self.embedding_model = FaceEmbedding(model_path=model_path)

    def get_embedding(self, img: np.ndarray) -> np.ndarray:
        return self.embedding_model.get_embedding(img)