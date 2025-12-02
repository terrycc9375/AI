from ultralytics import YOLO
import os
import yaml
import tempfile

os.environ['WANDB_MODE'] = 'disabled'

def dict_to_temp_yaml(cfg: dict):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.yaml')
    with open(tmp.name, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, sort_keys=False, allow_unicode=True)
    return tmp.name

# -------------------------------------------------------------
# 1.YOLO11原始架構
# -------------------------------------------------------------
yolo11_arch = {
    'nc': 1,
    'scales': {
        'n': [0.50, 0.25, 1024],
        's': [0.50, 0.50, 1024],
    },
    'backbone': [
        [-1, 1, 'Conv',   [64, 3, 2]],
        [-1, 1, 'Conv',   [128, 3, 2]],
        [-1, 2, 'C3k2',   [256, False, 0.25]],
        [-1, 1, 'Conv',   [256, 3, 2]],
        [-1, 2, 'C3k2',   [512, False, 0.25]],
        [-1, 1, 'Conv',   [512, 3, 2]],
        [-1, 2, 'C3k2',   [512, True]],
        [-1, 1, 'Conv',   [1024, 3, 2]],
        [-1, 2, 'C3k2',   [1024, True]],
        [-1, 1, 'SPPF',   [1024, 5]],
        [-1, 2, 'C2PSA',  [1024]],
    ],
    'head': [
        [-1, 1, 'nn.Upsample', [None, 2, "nearest"]],
        [[-1, 6], 1, 'Concat', [1]],
        [-1, 2, 'C3k2', [512, False]],

        [-1, 1, 'nn.Upsample', [None, 2, "nearest"]],
        [[-1, 4], 1, 'Concat', [1]],
        [-1, 2, 'C3k2', [256, False]],

        [-1, 1, 'Conv', [256, 3, 2]],
        [[-1, 13], 1, 'Concat', [1]],
        [-1, 2, 'C3k2', [512, False]],

        [-1, 1, 'Conv', [512, 3, 2]],
        [[-1, 10], 1, 'Concat', [1]],
        [-1, 2, 'C3k2', [1024, True]],

        [[16, 19, 22], 1, 'Detect', ['nc']]
    ]
}

# -------------------------------------------------------------
# 2.Dataset設定
# -------------------------------------------------------------
DATA_ROOT = r"請輸入你Dataset絕對路徑"  # ← 請改這裡(必要)
DATA_ROOT = DATA_ROOT.replace("\\", "/")

data_dict = {
    'train': f"{DATA_ROOT}/train/images",
    'val':   f"{DATA_ROOT}/test/images",
    'test':  f"{DATA_ROOT}/test/images",
    'nc': 1,
    'names': ['pig']
}

# -------------------------------------------------------------
# 3.YAML
# -------------------------------------------------------------
model_yaml_path = dict_to_temp_yaml(yolo11_arch)
data_yaml_path = dict_to_temp_yaml(data_dict)

# -------------------------------------------------------------
# 4.建立Model
# -------------------------------------------------------------
model = YOLO(model_yaml_path)

# -------------------------------------------------------------
# 5.Train
# -------------------------------------------------------------
results = model.train(
    data=data_yaml_path, #Dataset配置
    epochs=10, #Tain回合次數
    batch=16, #Batch Size (建議依你的GPU VRAM做調整)
    imgsz=320, #Input Image Resolution
    scale=0.5,
    mosaic=0.5,
    mixup=0.1,
    copy_paste=0.1,
    pretrained=False,
    #device="0",　＃有ＧＰＵ才需使用 (注意)
    workers=0,
    name="baseline_yolo11",
    save_period=10 #設定保存checkpoint
)

# -------------------------------------------------------------
# 6.Inference
# -------------------------------------------------------------
test_image = f"{DATA_ROOT}/test/images/00000001.jpg"
results = model(test_image)
results[0].show()
