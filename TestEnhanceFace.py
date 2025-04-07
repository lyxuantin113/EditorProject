from PIL import Image
import torch
from facenet_pytorch import MTCNN
from gfpgan import GFPGANer
import numpy as np
import cv2
import os

# ========== CẤU HÌNH ==========
INPUT_IMAGE = 'oldimg1.jpg'
OUTPUT_IMAGE = 'ImagesEnhanced/' + INPUT_IMAGE + '_enhanced.jpg'
GFPGAN_MODEL_PATH = 'GFPGAN/experiments/pretrained_models/GFPGANv1.4.pth'  # Đặt đường dẫn model tại đây

# ========== KHỞI TẠO ==========
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")

# Khởi tạo face detector
mtcnn = MTCNN(keep_all=True, device=device)

# Khởi tạo GFPGAN
gfpgan = GFPGANer(
    model_path=GFPGAN_MODEL_PATH,
    upscale=1,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None
)

# ========== ĐỌC ẢNH ==========
image = Image.open(INPUT_IMAGE).convert('RGB')
image_np = np.array(image)

# ========== PHÁT HIỆN KHUÔN MẶT ==========
boxes, _ = mtcnn.detect(image)

if boxes is None:
    print("Không tìm thấy khuôn mặt nào.")
    exit()

# ========== XỬ LÝ TỪNG KHUÔN MẶT ==========
for i, box in enumerate(boxes):
    pad = 20  # hoặc 30 tùy ảnh
    x1, y1 = max(0, int(box[0] - pad)), max(0, int(box[1] - pad))
    x2, y2 = min(image_np.shape[1], int(box[2] + pad)), min(image_np.shape[0], int(box[3] + pad))

    # Kiểm tra giới hạn
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image_np.shape[1], x2), min(image_np.shape[0], y2)

    # Crop khuôn mặt
    face_crop = image_np[y1:y2, x1:x2].copy()
    print(f"Processing face {i} at box: ({x1},{y1})-({x2},{y2}), size: {face_crop.shape}")
    cv2.imwrite(f"debug_face_{i}.png", cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))

    # Sử dụng GFPGAN để làm nét
    _, _, restored_face = gfpgan.enhance(
        face_crop, has_aligned=False, only_center_face=True, paste_back=True
    )

    if restored_face is None:
        print(f"⚠️ Không thể xử lý khuôn mặt tại box {i}. Bỏ qua...")
        continue

    # Resize lại cho khớp kích thước gốc
    enhanced_face = Image.fromarray(restored_face)
    enhanced_face = enhanced_face.resize((x2 - x1, y2 - y1))

    # Ghép lại ảnh gốc
    image.paste(enhanced_face, (x1, y1))

# ================= RESTORE TOÀN ẢNH =================
print("\n🧯 Đang khôi phục toàn bộ ảnh (restore full image)...")

# Dùng ảnh gốc (không crop), xử lý full face
_, _, restored_img = gfpgan.enhance(
    image_np, has_aligned=False, only_center_face=False, paste_back=False
)

# Lưu kết quả khôi phục toàn ảnh
if restored_img is not None and isinstance(restored_img, list) and len(restored_img) > 0:
    Image.fromarray(restored_img[0]).save('dx_restored_full.jpg')
    print("✔️ Ảnh đã được khôi phục toàn diện: dx_restored_full.jpg")
else:
    print("⚠️ Không thể khôi phục ảnh toàn diện.")


# ========== LƯU ẢNH ==========
image.save(OUTPUT_IMAGE)
print(f"✔️ Ảnh đã được lưu tại: {OUTPUT_IMAGE}")
