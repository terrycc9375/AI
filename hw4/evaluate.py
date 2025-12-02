import os
import yaml
import tempfile
from ultralytics import YOLO

def dict_to_temp_yaml(cfg: dict):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.yaml')
    with open(tmp.name, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, sort_keys=False, allow_unicode=True)
    return tmp.name

# ============================================================
# 1.Dataset設定
# ============================================================
DATA_ROOT = r"請輸入你Dataset絕對路徑"       # ← 請改這裡(必要)
DATA_ROOT = DATA_ROOT.replace("\\", "/")

data_dict = {
    'train': f"{DATA_ROOT}/train/images",
    'val':   f"{DATA_ROOT}/test/images",
    'test':  f"{DATA_ROOT}/test/images",
    'nc': 1,
    'names': ['pig']
}

data_yaml_path = dict_to_temp_yaml(data_dict)

# ============================================================
# 2.載入Model
# ============================================================
model = YOLO("best.pt")

# ============================================================
# 3.Test Set metrics計算
# ============================================================
metrics = model.val(
    data=data_yaml_path, 
    split="test",
    verbose=True,
    workers=0
)

ap50 = metrics.box.map50
mAP = metrics.box.map
precision = metrics.box.mp
recall = metrics.box.mr
f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

print("===== Test Results =====")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 score:  {f1:.4f}")
print(f"AP@50:     {ap50:.4f}")
print(f"mAP50-95:  {mAP:.4f}")

test_img_dir = data_dict["test"]

image_list = sorted([
    f for f in os.listdir(test_img_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

print(f"\nFound {len(image_list)} test images")
