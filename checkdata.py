# checkdata.py
import sys
import os
# Giả lập CLI args như khi bạn chạy train.py
sys.argv = ['E:/Senior_Student/Second_Term/ComputerVision/EditorProject/GFPGAN/gfpgan/train.py', '-opt', 'E:/Senior_Student/Second_Term/ComputerVision/EditorProject/GFPGAN/experiments/train_gfpgan_v1.4_finetune/train_gfpgan_v1.4_finetune.yml']

from basicsr.utils.options import parse_options
# Parse config
opt, _ = parse_options(root_path='.', is_train=True)

# Lấy path từ config
dataroot_lq = opt['datasets']['train']['dataroot_lq']
dataroot_gt = opt['datasets']['train']['dataroot_gt']

# In ra để kiểm tra
print('LQ folder:', dataroot_lq, '→ exists?', os.path.isdir(dataroot_lq))
print('GT folder:', dataroot_gt, '→ exists?', os.path.isdir(dataroot_gt))

from basicsr.data import build_dataset
ds = build_dataset(opt['datasets']['train'])
print('Num samples in train:', len(ds))
