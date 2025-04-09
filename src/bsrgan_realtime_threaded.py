import cv2
import threading
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# ====== Load Real-ESRGAN model ======
model_path = '../Real-ESRGAN/weights/RealESRGAN_x2plus.pth'
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
upscaler = RealESRGANer(
    scale=2,
    model_path=model_path,
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False
)

# ====== Global state ======
latest_frame = None
enhanced_frame = None
is_processing = False

# ====== Thread worker to process frame ======
def enhance_worker():
    global latest_frame, enhanced_frame, is_processing
    is_processing = True

    try:
        frame = latest_frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result, _ = upscaler.enhance(frame)
        enhanced_frame = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"❌ Lỗi khi xử lý frame: {e}")
        enhanced_frame = latest_frame

    is_processing = False

# ====== Main loop ======
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không mở được webcam!")
    exit()

print("📷 Bắt đầu Real-Time Enhancement với BSRGAN (multi-thread)...")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("Webcam (Raw)", frame)
    latest_frame = frame.copy()

    if not is_processing:
        threading.Thread(target=enhance_worker).start()

    if enhanced_frame is not None:
        cv2.imshow("Enhanced (BSRGAN)", enhanced_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()