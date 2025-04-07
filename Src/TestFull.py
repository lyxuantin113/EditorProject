import os
import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from gfpgan import GFPGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# ========== CẤU HÌNH ==========
INPUT_IMAGE = '../ImagesOrigin/dungXem.jpg'
OUTPUT_IMAGE = f'../ImagesEnhanced/{INPUT_IMAGE}_final.jpg'
GFPGAN_MODEL_PATH = '../GFPGAN/experiments/pretrained_models/GFPGANv1.4.pth'
REAL_ESRGAN_MODEL_PATH = '../Real-ESRGAN/weights/RealESRGAN_x2plus.pth'

# ========== HÀM CHUẨN HÓA ẢNH ==========
def ensure_rgb(image_pil):
    """
    Chuyển ảnh PIL thành np.ndarray RGB chuẩn, hỗ trợ ảnh grayscale, RGBA, hoặc đơn kênh.
    """
    try:
        img = image_pil.convert("RGB")
        img_np = np.array(img)

        if len(img_np.shape) == 2:
            img_np = np.expand_dims(img_np, axis=-1)

        if img_np.shape[-1] == 1:
            img_np = np.concatenate([img_np]*3, axis=-1)

        return img_np
    except Exception as e:
        print(f"⚠️ Lỗi khi chuẩn hóa ảnh: {e}")
        return None

# ========== LOAD Real-ESRGAN ==========
print("🔧 Đang nâng cấp background với Real-ESRGAN...")

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
upscaler = RealESRGANer(
    scale=2,
    model_path=REAL_ESRGAN_MODEL_PATH,
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False
)

img = Image.open(INPUT_IMAGE)
img_np = ensure_rgb(img)
if img_np is None:
    print("❌ Không thể xử lý ảnh đầu vào.")
    exit()

bg_enhanced, _ = upscaler.enhance(img_np)
print("✅ Background đã được xử lý.")

# ========== LOAD GFPGAN ==========
print("🤖 Đang phục hồi khuôn mặt với GFPGAN...")

gfpgan = GFPGANer(
    model_path=GFPGAN_MODEL_PATH,
    upscale=1,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"📡 Sử dụng thiết bị: {device}")
mtcnn = MTCNN(keep_all=True, device=device)

# Convert sang PIL và detect khuôn mặt
image = Image.fromarray(bg_enhanced)
image_np = np.array(image)

boxes, _ = mtcnn.detect(image)
if boxes is None:
    print("❌ Không tìm thấy khuôn mặt nào.")
    exit()

# ========== XỬ LÝ TỪNG KHUÔN MẶT ==========
for i, box in enumerate(boxes):
    pad = 20
    x1, y1 = max(0, int(box[0] - pad)), max(0, int(box[1] - pad))
    x2, y2 = min(image_np.shape[1], int(box[2] + pad)), min(image_np.shape[0], int(box[3] + pad))
    face_crop = image_np[y1:y2, x1:x2].copy()

    _, _, restored_face = gfpgan.enhance(
        face_crop, has_aligned=False, only_center_face=True, paste_back=True
    )

    if restored_face is None:
        print(f"⚠️ Không thể xử lý khuôn mặt tại box {i}. Bỏ qua...")
        continue

    enhanced_face = Image.fromarray(restored_face).resize((x2 - x1, y2 - y1))
    image.paste(enhanced_face, (x1, y1))

# ========== LƯU ẢNH ==========
os.makedirs("../ImagesEnhanced", exist_ok=True)
image.save(OUTPUT_IMAGE)
print(f"✅ Ảnh đã được xử lý hoàn chỉnh và lưu tại: {OUTPUT_IMAGE}")