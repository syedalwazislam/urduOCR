import torch
import io
import base64
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from read import text_recognizer
from model import Model
from utils import CTCLabelConverter
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display
import cv2


# Check for required files
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

""" vocab / character number configuration """
file = open("UrduGlyphs.txt", "r", encoding="utf-8")
content = file.readlines()
content = ''.join([str(elem).strip('\n') for elem in content])
content = content + " "

""" model configuration """
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


app = FastAPI(
    title="End-to-End Urdu OCR API",
    description="API for UTRNet - High-Resolution Urdu Text Recognition"
)

# Enable CORS for external frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def fix_urdu_text(extracted_text):
    """
    Fix Urdu text rendering by reshaping Arabic/Urdu characters and applying RTL direction.

    Args:
        extracted_text: Raw text extracted from OCR (may have disconnected letters)

    Returns:
        Properly formatted Urdu text with correct RTL rendering
    """
    if not extracted_text or not extracted_text.strip():
        return extracted_text

    try:
        # Step 1: Reshape Arabic/Urdu characters (connect them properly)
        reshaped_text = arabic_reshaper.reshape(extracted_text)

        # Step 2: Apply right-to-left direction
        bidi_text = get_display(reshaped_text)

        return bidi_text
    except Exception as e:
        # If reshaping fails, return original text
        print(f"Warning: Failed to reshape Urdu text: {e}")
        return extracted_text


# ---------------------------------------------------------------------------
# Phrases that should never appear in the OCR output (e.g. printed footers).
# Matching is done on the raw (pre-bidi) text so it is encoding-independent.
# Add any additional unwanted phrases to this list as needed.
# ---------------------------------------------------------------------------
BLOCKED_PHRASES = [
    "گمشدہ کارڈ ملنے پر قریبی لیٹر بکس میں ڈال دیں",
    "کارڈ ملنے پر",
    "لیٹر بکس میں ڈال دیں",
    "ڈال دیں",
]

# Fraction of image height from the bottom that is treated as the footer zone.
# Any bounding-box whose TOP edge falls below this threshold is discarded
# before phrase matching, giving a second independent layer of protection.
# Adjust this value if the footer sits higher up on the page (e.g. 0.85).
FOOTER_ZONE_FRACTION = 0.90  # bottom 10% of the image


def _is_footer_box(box, image_height: int) -> bool:
    """Return True if the bounding box sits in the footer zone."""
    _x1, y1, _x2, _y2 = box
    return y1 > image_height * FOOTER_ZONE_FRACTION


def _is_blocked_text(text: str) -> bool:
    """Return True if the recognised text matches any blocked phrase."""
    for phrase in BLOCKED_PHRASES:
        if phrase in text:
            return True
    return False


def predict(input_image: Image.Image, return_annotated_image: bool = False):
    """Line Detection"""
    detection_results = detection_model.predict(
        source=input_image, conf=0.2, imgsz=1280, save=False, nms=True, device=device
    )
    bounding_boxes = detection_results[0].boxes.xyxy.cpu().numpy().tolist()
    bounding_boxes.sort(key=lambda x: x[1])

    # ------------------------------------------------------------------
    # Robust footer removal — Layer 1: spatial filter
    # Drop any box whose top edge is in the bottom FOOTER_ZONE_FRACTION
    # of the image (e.g. the last 10% of image height).
    # ------------------------------------------------------------------
    image_height = input_image.size[1]
    bounding_boxes = [
        box for box in bounding_boxes
        if not _is_footer_box(box, image_height)
    ]

    """Draw the bounding boxes"""
    annotated_image = input_image.copy()
    draw = ImageDraw.Draw(annotated_image)
    for box in bounding_boxes:
        # draw rectangle outline with random color and width=5
        draw.rectangle(box, fill=None, outline=tuple(np.random.randint(0, 255, 3)), width=5)

    """Crop the detected lines"""
    cropped_images = []
    for box in bounding_boxes:
        cropped_images.append(input_image.crop(box))

    """Recognize the text"""
    texts = []
    for img in cropped_images:
        raw_text = text_recognizer(img, recognition_model, converter, device)

        # ------------------------------------------------------------------
        # Robust footer removal — Layer 2: phrase blocklist filter
        # Even if a box slipped through the spatial filter, block it here.
        # ------------------------------------------------------------------
        if _is_blocked_text(raw_text):
            print(f"[filter] Blocked line: {raw_text!r}")
            continue

        # Fix Urdu text rendering (reshape and apply RTL)
        fixed_text = fix_urdu_text(raw_text)
        texts.append(fixed_text)

    """Join the text"""
    text = "\n".join(texts)

    result = {
        "text": text,
        "lines_detected": len(bounding_boxes),
        "bounding_boxes": bounding_boxes
    }

    if return_annotated_image:
        # Convert annotated image to base64
        img_buffer = io.BytesIO()
        annotated_image.save(img_buffer, format="PNG")
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        result["annotated_image"] = f"data:image/png;base64,{img_base64}"

    return result


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
    - **return_annotated_image**: If True, returns base64-encoded annotated image with bounding boxes

    Returns JSON with recognized text and optional annotated image.
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read and convert uploaded file to PIL Image
        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Process the image
        result = predict(input_image, return_annotated_image=return_annotated_image)

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)