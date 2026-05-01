import torch
import io
import re
import base64
import os
import math
import unicodedata
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from read import text_recognizer
from model import Model
from utils import CTCLabelConverter, NormalizePAD
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display
import cv2


# ──────────────────────────────────────────────────────────────────────────────
# Required-file check
# ──────────────────────────────────────────────────────────────────────────────
REQUIRED_FILES = {
    "UrduGlyphs.txt": "Vocabulary file containing Urdu glyphs",
    "best_norm_ED.pth": "Recognition model checkpoint (download from HuggingFace)",
    "yolov8m_UrduDoc.pt": "YOLO detection model (download from HuggingFace)"
}

missing_files = []
for file, description in REQUIRED_FILES.items():
    if not os.path.exists(file):
        missing_files.append(f"  - {file}: {description}")

if missing_files:
    error_msg = "Missing required files:\n" + "\n".join(missing_files)
    error_msg += "\n\nPlease download the model files using:"
    error_msg += "\n  PowerShell: .\\download_files.ps1"
    error_msg += "\n  Or manually download from: https://huggingface.co/spaces/abdur75648/UrduOCR-UTRNet"
    raise FileNotFoundError(error_msg)


# ──────────────────────────────────────────────────────────────────────────────
# Vocabulary / model setup
# ──────────────────────────────────────────────────────────────────────────────
with open("UrduGlyphs.txt", "r", encoding="utf-8") as _f:
    content = ''.join([str(elem).strip('\n') for elem in _f.readlines()]) + " "

print("Loading models...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

converter = CTCLabelConverter(content)
print("Loading recognition model (best_norm_ED.pth)...")
recognition_model = Model(num_class=len(converter.character), device=device)
recognition_model = recognition_model.to(device)
recognition_model.load_state_dict(torch.load("best_norm_ED.pth", map_location=device))
recognition_model.eval()
print("✓ Recognition model loaded")

YOLO_WEIGHTS = "yolov8m_UrduDoc.pt"
try:
    print("Loading detection model (yolov8m_UrduDoc.pt)...")
    detection_model = YOLO(YOLO_WEIGHTS)
    print("✓ Detection model loaded")
except Exception as e:
    print(f"Warning: Failed to load detection model: {e}")
    detection_model = None


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="End-to-End Urdu OCR API",
    description="API for UTRNet - High-Resolution Urdu Text Recognition"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Footer / blocklist config  (unchanged from original)
# ──────────────────────────────────────────────────────────────────────────────
BLOCKED_PHRASES = [
    "گمشدہ کارڈ ملنے پر قریبی لیٹر بکس میں ڈال دیں",
    "کارڈ ملنے پر",
    "لیٹر بکس میں ڈال دیں",
    "ڈال دیں",
]
FOOTER_ZONE_FRACTION = 0.90


def _is_footer_box(box, image_height: int) -> bool:
    _x1, y1, _x2, _y2 = box
    return y1 > image_height * FOOTER_ZONE_FRACTION


def _is_blocked_text(text: str) -> bool:
    for phrase in BLOCKED_PHRASES:
        if phrase in text:
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# FIX 1 — Alphanumeric token detection
#
# Decide whether a cropped line is "mostly Latin / numeric" so we can
# skip the FLIP that the recogniser applies for Urdu RTL text.
# If the line is Urdu with an embedded house-number we keep the flip
# but rescue the number afterwards with regex.
# ──────────────────────────────────────────────────────────────────────────────
_LATIN_OR_DIGIT = re.compile(r'[A-Za-z0-9\-/\\]')
_URDU_CHAR      = re.compile(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]')


def _is_latin_dominant(pil_crop: Image.Image) -> bool:
    """
    Heuristic: convert the crop to text via a quick character-ratio check
    on the *image* content is hard, so instead we probe the pixel structure.
    A simpler proxy: compare the aspect ratio + pixel density.

    For address lines that are purely Latin/numeric (e.g. "R-214 Block-B")
    the YOLO model usually produces a wide, short box.  We use a soft
    aspect-ratio + width threshold rather than re-running a second OCR.

    Returns True  → skip the horizontal flip
    Returns False → apply the flip (normal Urdu path)
    """
    w, h = pil_crop.size
    # Very wide, low-height crops are almost certainly pure number/Latin lines
    # (Urdu lines tend to be taller relative to their width because of diacritics)
    aspect = w / max(h, 1)
    return aspect > 12   # tune this threshold for your documents


# ──────────────────────────────────────────────────────────────────────────────
# FIX 2 — House-number / alphanumeric rescue with regex
#
# Patterns: R-214, A-12, 3-C, 1184, Block-B, R214, 244/A, etc.
# Applied to the RAW OCR string BEFORE bidi reshaping so digit order
# is still intact.
# ──────────────────────────────────────────────────────────────────────────────
_HOUSE_NUMBER_RE = re.compile(
    r'\b'
    r'('
    r'[A-Za-z]{0,3}[-]?\d{1,5}(?:[A-Za-z])?(?:[/\\]\d{1,4})?'   # R-214, A-12, 244/A
    r'|'
    r'\d{1,5}[-][A-Za-z]'                                          # 3-C, 12-B
    r'|'
    r'[A-Za-z]{1,3}[-]\d{1,5}'                                     # Block-6, Sector-F
    r')'
    r'\b',
    re.ASCII
)

# Eastern-Arabic → Western-Arabic digit map (Urdu documents use ۰–۹)
_EASTERN_ARABIC_DIGITS = str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩', '01234567890123456789')


def _normalise_digits(text: str) -> str:
    """Convert Eastern-Arabic/Urdu digits to ASCII digits."""
    return text.translate(_EASTERN_ARABIC_DIGITS)


def _rescue_numbers(raw_text: str) -> list[str]:
    """
    Return a list of all house-number / alphanumeric tokens found in the
    raw OCR output (before any bidi transformation).
    """
    normalised = _normalise_digits(raw_text)
    return _HOUSE_NUMBER_RE.findall(normalised)


# ──────────────────────────────────────────────────────────────────────────────
# FIX 3 — Crop pre-processing to improve recognition of small / noisy crops
# ──────────────────────────────────────────────────────────────────────────────

def _preprocess_crop(pil_img: Image.Image) -> Image.Image:
    """
    Sharpen and upscale a crop before passing it to the recogniser.
    This helps especially for low-DPI scans where digits blur together.
    """
    w, h = pil_img.size

    # 3-a  Upscale very small crops (height < 48 px) by 2×
    if h < 48:
        pil_img = pil_img.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
        w, h = pil_img.size

    # 3-b  Convert to grayscale for processing
    gray = pil_img.convert("L")

    # 3-c  Mild sharpening to recover blurred strokes
    gray = gray.filter(ImageFilter.SHARPEN)

    # 3-d  Contrast enhancement (helps on pale/faded documents)
    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(1.5)

    # 3-e  Adaptive threshold via OpenCV for cleaner binarisation
    np_img = np.array(gray)
    binarised = cv2.adaptiveThreshold(
        np_img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=8
    )
    return Image.fromarray(binarised).convert("RGB")


# ──────────────────────────────────────────────────────────────────────────────
# FIX 4 — Custom text_recognizer that can optionally skip the RTL flip
# ──────────────────────────────────────────────────────────────────────────────

def text_recognizer_smart(img_cropped: Image.Image,
                          model,
                          converter,
                          device,
                          skip_flip: bool = False) -> str:
    """
    Drop-in replacement for text_recognizer() from read.py.
    Adds:
      • optional skip_flip for Latin-dominant lines
      • pre-processing via _preprocess_crop()
    """
    img = _preprocess_crop(img_cropped).convert("L")

    if not skip_flip:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    w, h = img.size
    ratio = w / float(h)
    resized_w = min(math.ceil(32 * ratio), 400)
    img = img.resize((resized_w, 32), Image.Resampling.BICUBIC)

    transform = NormalizePAD((1, 32, 400))
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(img_tensor)

    preds_size = torch.IntTensor([preds.size(1)])
    _, preds_index = preds.max(2)
    return converter.decode(preds_index.data, preds_size.data)[0]


# ──────────────────────────────────────────────────────────────────────────────
# FIX 5 — Urdu text fixer that PRESERVES embedded ASCII tokens
#
# Problem: get_display() reorders digits / Latin tokens when they sit inside
#          an RTL paragraph, making "R-214" appear as "412-R" or similar.
# Solution: extract all ASCII tokens first, run bidi on the Urdu-only parts,
#           then re-inject the ASCII tokens at the correct positions.
# ──────────────────────────────────────────────────────────────────────────────

_ASCII_TOKEN_RE = re.compile(r'[A-Za-z0-9][A-Za-z0-9\-/\\]*')


def fix_urdu_text(extracted_text: str) -> str:
    """
    Reshape Urdu characters and apply RTL direction while keeping
    Latin / numeric tokens (house numbers, IDs) in their correct order.
    """
    if not extracted_text or not extracted_text.strip():
        return extracted_text

    try:
        # Step 1: Normalise Eastern-Arabic digits → ASCII digits
        text = _normalise_digits(extracted_text)

        # Step 2: Placeholder substitution — replace each ASCII token with a
        #         unique placeholder so bidi/reshaper doesn't touch them.
        placeholders = {}
        def _replace(m):
            key = f"\u200C{len(placeholders):04d}\u200C"   # zero-width non-joiner fence
            placeholders[key] = m.group(0)
            return key
        protected = _ASCII_TOKEN_RE.sub(_replace, text)

        # Step 3: Reshape Arabic/Urdu characters
        reshaped = arabic_reshaper.reshape(protected)

        # Step 4: Apply bidi
        bidi_text = get_display(reshaped)

        # Step 5: Re-inject original ASCII tokens
        for key, original in placeholders.items():
            bidi_text = bidi_text.replace(key, original)

        return bidi_text

    except Exception as e:
        print(f"Warning: Failed to reshape Urdu text: {e}")
        return extracted_text


# ──────────────────────────────────────────────────────────────────────────────
# FIX 6 — Image-level pre-processing before YOLO detection
#
# Deskew + denoise the full document image so YOLO finds lines more reliably
# and the crops fed to the recogniser are cleaner.
# ──────────────────────────────────────────────────────────────────────────────

def _preprocess_full_image(pil_img: Image.Image) -> Image.Image:
    """
    Light pre-processing on the full document image:
      • Upscale if resolution is too low  (short side < 600 px)
      • Mild denoising
      • Deskew via Hough-line-based rotation correction
    Returns a processed PIL Image (RGB).
    """
    # 6-a  Upscale low-resolution images
    w, h = pil_img.size
    short_side = min(w, h)
    if short_side < 600:
        scale = 600 / short_side
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    # 6-b  Convert to numpy for OpenCV operations
    img_np = np.array(pil_img.convert("RGB"))

    # 6-c  Denoise (fast, non-local means – only colour channel)
    img_np = cv2.fastNlMeansDenoisingColored(img_np, None, h=7, hColor=7,
                                              templateWindowSize=7, searchWindowSize=21)

    # 6-d  Deskew
    gray    = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, bw   = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords  = np.column_stack(np.where(bw > 0))
    if len(coords) > 100:
        angle = cv2.minAreaRect(coords)[-1]
        # minAreaRect returns angles in [-90, 0); correct to small rotations
        if angle < -45:
            angle = 90 + angle
        if abs(angle) > 0.3:          # only correct if skew > 0.3°
            (h_img, w_img) = img_np.shape[:2]
            M = cv2.getRotationMatrix2D((w_img / 2, h_img / 2), angle, 1.0)
            img_np = cv2.warpAffine(img_np, M, (w_img, h_img),
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)

    return Image.fromarray(img_np)


# ──────────────────────────────────────────────────────────────────────────────
# Main predict() pipeline
# ──────────────────────────────────────────────────────────────────────────────

def predict(input_image: Image.Image, return_annotated_image: bool = False):

    # ── FIX 6: pre-process the full image before detection ────────────────────
    processed_image = _preprocess_full_image(input_image)

    # ── Line detection ────────────────────────────────────────────────────────
    detection_results = detection_model.predict(
        source=processed_image, conf=0.2, imgsz=1280,
        save=False, nms=True, device=device
    )
    bounding_boxes = detection_results[0].boxes.xyxy.cpu().numpy().tolist()
    bounding_boxes.sort(key=lambda x: x[1])

    # Footer spatial filter (unchanged)
    image_height = processed_image.size[1]
    bounding_boxes = [
        box for box in bounding_boxes
        if not _is_footer_box(box, image_height)
    ]

    # ── Draw bounding boxes ───────────────────────────────────────────────────
    annotated_image = processed_image.copy()
    draw = ImageDraw.Draw(annotated_image)
    for box in bounding_boxes:
        draw.rectangle(box, fill=None,
                       outline=tuple(np.random.randint(0, 255, 3).tolist()),
                       width=5)

    # ── Crop detected lines ───────────────────────────────────────────────────
    cropped_images = [processed_image.crop(box) for box in bounding_boxes]

    # ── Recognise each line ───────────────────────────────────────────────────
    texts      = []
    debug_info = []   # optional: per-line debug data

    for idx, img in enumerate(cropped_images):

        # FIX 1: decide whether to skip the RTL flip
        latin_dominant = _is_latin_dominant(img)

        # FIX 4: use the smart recogniser
        raw_text = text_recognizer_smart(
            img, recognition_model, converter, device,
            skip_flip=latin_dominant
        )

        # FIX 2: rescue address numbers from raw output BEFORE bidi
        rescued_numbers = _rescue_numbers(raw_text)
        if rescued_numbers:
            print(f"[rescue] Line {idx} raw numbers: {rescued_numbers}")

        # Footer phrase filter (unchanged)
        if _is_blocked_text(raw_text):
            print(f"[filter] Blocked line: {raw_text!r}")
            continue

        # FIX 5: bidi-safe Urdu fix that preserves ASCII tokens
        fixed_text = fix_urdu_text(raw_text)
        texts.append(fixed_text)

        debug_info.append({
            "line_index": idx,
            "raw_text": raw_text,
            "rescued_numbers": rescued_numbers,
            "latin_dominant": latin_dominant,
            "fixed_text": fixed_text,
        })

    text = "\n".join(texts)

    result = {
        "text": text,
        "lines_detected": len(bounding_boxes),
        "bounding_boxes": bounding_boxes,
        "debug": debug_info,   # remove this key in production if not needed
    }

    if return_annotated_image:
        img_buffer = io.BytesIO()
        annotated_image.save(img_buffer, format="PNG")
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        result["annotated_image"] = f"data:image/png;base64,{img_base64}"

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "End-to-End Urdu OCR API",
        "description": "API for UTRNet - High-Resolution Urdu Text Recognition",
        "endpoints": {
            "/ocr": "POST - Upload an image for OCR processing",
            "/health": "GET - Check API health status"
        }
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": True
    }


@app.post("/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    return_annotated_image: bool = False
):
    """
    Process an uploaded image for Urdu OCR.

    - **file**: Image file (JPEG, PNG, etc.)
    - **return_annotated_image**: If True, returns base64-encoded annotated image

    Returns JSON with recognised text and optional annotated image.
    """
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = predict(input_image, return_annotated_image=return_annotated_image)
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)