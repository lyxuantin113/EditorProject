import cv2
import numpy as np
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from PIL import Image

# ========== CẤU HÌNH ==========
model_path = 'Real-ESRGAN/weights/RealESRGAN_x2plus.pth'
input_path = 'oldimg1.jpg.jpg'
output_path = f'ImagesEnhanced/{input_path}_enhanced_esrgan.jpg'

# ========== LOAD MÔ HÌNH ==========
model = RRDBNet(
    num_in_ch=3, num_out_ch=3,
    num_feat=64, num_block=23,
    num_grow_ch=32, scale=2
)

upscaler = RealESRGANer(
    scale=2,
    model_path=model_path,
    model=model,
    tile=0,  # nếu GPU yếu có thể dùng tile=128
    tile_pad=10,
    pre_pad=0,
    half=False  # nếu dùng GPU và muốn tiết kiệm VRAM, chuyển True
)

# ========== XỬ LÝ ẢNH ==========
img = cv2.imread(input_path, cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

output, _ = upscaler.enhance(img)

# ========== LƯU ẢNH ==========
Image.fromarray(output).save(output_path)
print(f"✅ Ảnh đã được làm rõ và lưu tại: {output_path}")