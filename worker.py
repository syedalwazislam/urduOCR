import io
import base64
import torch
import numpy as np
from PIL import Image, ImageDraw
import redis
from rq import Worker, Queue
from rq import Connection

from ultralytics import YOLO
from model import Model
from utils import CTCLabelConverter
from read import text_recognizer

import arabic_reshaper
from bidi.algorithm import get_display

# 🔗 Redis connection
redis_conn = redis.Redis(host="redis", port=6379)
queue = Queue("ocr_queue", connection=redis_conn)

# 🔥 Load models ONCE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocab
with open("UrduGlyphs.txt", "r", encoding="utf-8") as f:
    content = "".join([line.strip() for line in f]) + " "

converter = CTCLabelConverter(content)

recognition_model = Model(num_class=len(converter.character), device=device)
recognition_model.load_state_dict(torch.load("best_norm_ED.pth", map_location=device))
recognition_model.to(device)
recognition_model.eval()

# YOLO
detection_model = YOLO("yolov8m_UrduDoc.pt")

print("✅ Models Loaded Successfully")

# 🔧 Fix Urdu text
def fix_urdu_text(text):
    try:
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except:
        return text


# 🔥 Core OCR pipeline
def predict(image: Image.Image):
    results = detection_model.predict(
        source=image,
        conf=0.2,
        imgsz=640,  # optimized
        device=device,
        save=False
    )

    boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
    boxes.sort(key=lambda x: x[1])

    texts = []

    for box in boxes:
        cropped = image.crop(box)
        raw_text = text_recognizer(cropped, recognition_model, converter, device)
        texts.append(fix_urdu_text(raw_text))

    return "\n".join(texts)


# 🔥 Worker job function
def process_ocr_job(img_base64):
    try:
        img_bytes = base64.b64decode(img_base64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        text = predict(image)

        return {
            "text": text
        }

    except Exception as e:
        return {
            "error": str(e)
        }


# 🚀 Start worker
if __name__ == "__main__":
    with Connection(redis_conn):
        worker = Worker(["ocr_queue"])
        worker.work()