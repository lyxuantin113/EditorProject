import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os
import cv2
import numpy as np
import torch
from PIL import Image
from io import BytesIO
from gfpgan import GFPGANer
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import uuid

# ========== KHỞI TẠO FASTAPI ==========
app = FastAPI(title="GFPGAN API (Image + Video)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép frontend truy cập
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== LOAD GFPGAN ==========
model_path = "GFPGAN/experiments/pretrained_models/GFPGANv1.4.pth"
restorer = GFPGANer(
    model_path=model_path,
    upscale=2,
    arch="clean",
    channel_multiplier=2,
    bg_upsampler=None
)

# ========== API: Phục hồi ảnh ==========
@app.post("/restore")
async def restore_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    image = np.array(image)

    _, _, restored_image = restorer.enhance(
        image,
        has_aligned=False,
        only_center_face=False,
        paste_back=True
    )

    output_path = f"restored_image_{uuid.uuid4().hex}.jpg"
    Image.fromarray(restored_image).save(output_path)

    return FileResponse(output_path, media_type="image/jpeg")


# ========== API: Phục hồi video ==========
@app.post("/enhance-video/")
async def enhance_video(file: UploadFile = File(...)):
    # Bước 1: Lưu file tạm
    input_name = f"temp_{uuid.uuid4().hex}.mp4"
    with open(input_name, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(input_name)
    if not cap.isOpened():
        return {"error": "❌ Không mở được video."}

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Bước 2: Tạo file .avi tạm
    temp_avi = input_name.replace(".mp4", "_enhanced.avi")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(temp_avi, fourcc, fps, (frame_w, frame_h))

    # Bước 3: Xử lý từng frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, _, restored = restorer.enhance(
                rgb,
                has_aligned=False,
                only_center_face=True,
                paste_back=True
            )
            if isinstance(restored, list):
                restored = restored[0]

            bgr = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
            if bgr.shape[:2] != (frame_h, frame_w):
                bgr = cv2.resize(bgr, (frame_w, frame_h))
        except Exception as e:
            print(f"⚠️ Lỗi frame: {e}")
            bgr = frame

        out.write(bgr)

    cap.release()
    out.release()

    # Bước 4: Chuyển sang .mp4 bằng ffmpeg
    final_output = input_name.replace(".mp4", "_final.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-i", temp_avi,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-movflags", "+faststart",
        final_output
    ])

    # Cleanup
    os.remove(input_name)
    os.remove(temp_avi)

    return FileResponse(final_output, media_type="video/mp4", filename="enhanced_video.mp4")

# uvicorn main:app --host 0.0.0.0 --port 8000