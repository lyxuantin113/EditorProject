import os
import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from gfpgan import GFPGANer

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'BSRGAN')))
from utils import utils_image as util
from models.network_rrdbnet import RRDBNet as net

# ========== CẤU HÌNH ==========
INPUT_IMAGE = '../ImagesOrigin/noise1.jpg'
OUTPUT_IMAGE = f'../ImagesEnhanced/{os.path.basename(INPUT_IMAGE)}_bsrgfp.jpg'
GFPGAN_MODEL_PATH = '../GFPGAN/experiments/pretrained_models/GFPGANv1.4.pth'
BSRGAN_MODEL_PATH = '../BSRGAN/model_zoo/BSRGAN.pth'

# ========== HÀM CHUẨN HÓA ẢNH ==========
def ensure_rgb(image_pil):
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

# ========== LOAD BSRGAN ==========
print("🔧 Đang nâng cấp toàn ảnh với BSRGAN...")

model = net(in_nc=3, out_nc=3, nf=64, nb=23)
model.load_state_dict(torch.load(BSRGAN_MODEL_PATH), strict=True)
model.eval()
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

img = Image.open(INPUT_IMAGE)
img_np = ensure_rgb(img)
if img_np is None:
    print("❌ Không thể xử lý ảnh đầu vào.")
    exit()

# Chuẩn hóa và chuyển sang tensor
img_tensor = util.uint2tensor4(img_np).to(device)
with torch.no_grad():
    output_tensor = model(img_tensor)
output_np = util.tensor2uint(output_tensor)

print("✅ Ảnh đã được nâng cấp bằng BSRGAN.")

# ========== LOAD GFPGAN ==========
print("🤖 Đang phục hồi khuôn mặt với GFPGAN...")

gfpgan = GFPGANer(
    model_path=GFPGAN_MODEL_PATH,
    upscale=1,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None
)

mtcnn = MTCNN(keep_all=True, device=device)

# Convert sang PIL và detect khuôn mặt
image = Image.fromarray(output_np)
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