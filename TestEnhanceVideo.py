import cv2
import numpy as np
from gfpgan import GFPGANer
from PIL import Image

# ========== CẤU HÌNH ==========
video_path = 'datasets/v2.mp4'  # Video đầu vào
output_path = 'datasets/v2_enhanced.mp4'  # Video đầu ra
GFPGAN_MODEL_PATH = 'GFPGAN/experiments/pretrained_models/GFPGANv1.4.pth'

# ========== KHỞI TẠO ==========
gfpgan = GFPGANer(
    model_path=GFPGAN_MODEL_PATH,
    upscale=1,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None
)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ Không mở được video: {video_path}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"🔄 FPS: {fps}, Kích thước: {frame_w}x{frame_h}")

# Cấu hình codec để xuất ra .mp4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    print(f"🔧 Đang xử lý frame {frame_idx}...")

    try:
        # Convert BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Chạy GFPGAN
        _, _, restored_frame = gfpgan.enhance(
            frame_rgb,
            has_aligned=False,
            only_center_face=True,
            paste_back=True
        )

        # Nếu trả về list thì lấy frame đầu
        if isinstance(restored_frame, list):
            restored_frame = restored_frame[0]

        # Convert RGB → BGR để ghi video
        frame_bgr = cv2.cvtColor(restored_frame, cv2.COLOR_RGB2BGR)

        # Resize lại nếu không khớp (phòng lỗi ghi)
        if frame_bgr.shape[1] != frame_w or frame_bgr.shape[0] != frame_h:
            frame_bgr = cv2.resize(frame_bgr, (frame_w, frame_h))

    except Exception as e:
        print(f"⚠️ Lỗi tại frame {frame_idx}: {e}")
        frame_bgr = frame  # Dùng frame gốc nếu lỗi

    out.write(frame_bgr)

cap.release()
out.release()
print(f"✅ Video đã xử lý xong và lưu tại: {output_path}")
