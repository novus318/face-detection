# Face Verification API

Fast face verification with document intelligence for KYC applications.


## Quick Start

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```


## API Endpoints

### Verify Faces

```bash
POST /api/v1/verify
```

```json
{
  "m1_url": "https://example.com/passport.jpg",
  "m2_urls": ["https://example.com/selfie.jpg"]
}
```

### Document Intelligence (ID Analysis)

```bash
POST /api/v1/doc-intel
```

```json
{
  "document_url": "https://example.com/id.jpg"
}
```

Use `?use_sample=true` to test with sample image.


## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/yolov5n-face.onnx` | ONNX model path |
| `IMG_SIZE` | `320` | Detection input size |
| `CONFIDENCE_THRESHOLD` | `0.45` | Face detection threshold |
| `MAX_IMAGE_SIZE` | `1024` | Max image dimension |
| `DOWNLOAD_TIMEOUT` | `5` | Download timeout (seconds) |
| `AZURE_DOC_INTEL_ENDPOINT` | - | Azure Document Intelligence endpoint |
| `AZURE_DOC_INTEL_KEY` | - | Azure API key |


## Project Structure

```
app/
├── main.py              # FastAPI app
├── config.py            # Settings
├── api/v1/
│   ├── verify.py        # POST /api/v1/verify
│   └── doc_intel.py     # POST /api/v1/doc-intel
└── core/
    ├── models.py        # ONNX detector
    ├── alignment.py     # Face alignment
    ├── matcher.py       # Face matching
    └── downloader.py    # Async image fetcher
models/
└── yolov5n-face.onnx
```


## Performance

- Cold start: ~3-5s
- Warm response: 1-3s
- Face detection: ~100ms
- Face embedding: ~50ms