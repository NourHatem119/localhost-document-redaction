# image_vision — Offline Image PII Detection & Redaction

Pure computer-vision pipeline for detecting and redacting PII in images. No cloud services or network access required.

## Detected labels

- `FACE` — detected via OpenCV Haar cascade (always works offline)
- `TEXT_PII` — text blocks detected via Tesseract OCR (falls back to mock regions if Tesseract is not installed)
- `ID` — card-like rectangular regions detected via contour heuristics
- `SIGNATURE` — elongated low-text-density regions detected via contour heuristics

## Windows Setup

### 1. Install Tesseract OCR (optional but recommended)

1. Download the installer from [UB Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki).
2. Install to the default location: `C:\Program Files\Tesseract-OCR\tesseract.exe`.
3. The pipeline will auto-detect this path. You can also set the environment variable:
   ```powershell
   $env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
   ```

If Tesseract is not installed, the pipeline falls back to mock `TEXT_PII` regions so the demo still runs.

### 2. Install Python dependencies

```bash
pip install opencv-python pytesseract
```

`mediapipe` is listed in the spec but not currently used by the offline detectors; it can be installed if future face-detection improvements are added.

### 3. Run the demo

```bash
python -m image_vision.demo
# or
python image_vision/demo.py
```

The demo:
- Creates two synthetic test images (`test_images/face_text.png` and `test_images/id_signature.png`) if they don't exist.
- Runs `detect_image_pii` on each image.
- Prints the detected `ImageRegion` list.
- Runs `redact_image` and writes output to `redacted_output/`.

## API

```python
from image_vision import detect_image_pii, redact_image, is_tesseract_available

regions = detect_image_pii("scan.png", doc_id="doc_1", image_id="img_0", page_no=0)
print(regions)  # List[ImageRegion]

output_path = redact_image("scan.png", regions, "redacted.png")
```

## Architecture

- `detector.py` — `detect_image_pii` orchestrates face, text, and heuristic detectors.
- `redaction.py` — `redact_image` applies Gaussian blur (51×51) + pixelation per region.
- `demo.py` — standalone demo script with synthetic image generation.
- `schema.py` (repo root) — shared `ImageRegion`, `BBox`, `ImageLabel`, and `ImageSource` types.
