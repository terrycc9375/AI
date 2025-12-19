from ultralytics.models.yolo.model import YOLO
from ultralytics.engine.trainer import BaseTrainer
import os
import time
import argparse
import yaml
import tempfile

# Data Augmentation
import albumentations as A

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def dict_to_temp_yaml(cfg: dict):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.yaml')
    with open(tmp.name, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, sort_keys=False, allow_unicode=True)
    return tmp.name

def train(
    epochs: int = 10,
):
    model = YOLO("yolo11s.pt")
    root = r"D:\NYCU\AI\hw4\dataset"
    data = {
		"nc": 1,
		"names": ["pig"],
		"train": os.path.join(root, "images\\train"),
		"val": os.path.join(root, "images\\test"),
	}
    yaml_data = dict_to_temp_yaml(data)

    aug = A.Compose(
        [
            A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Blur(blur_limit=5, p=0.3),
        ],
        bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
    )
    
    # richcb = RichProgressCallback()
    # model.add_callback("on_train_start", richcb.on_train_start)
    # model.add_callback("on_train_epoch_start", richcb.on_train_epoch_start)
    # model.add_callback("on_train_batch_end", richcb.on_train_batch_end)
    # model.add_callback("on_train_batch_end", richcb.on_train_batch_end)
    # model.add_callback("on_train_end", richcb.on_train_end)
    
    training_logs = model.train(
		data=yaml_data,
		epochs=epochs,
		batch=4,
		imgsz=1280,
		project="YOLO11",
		name="ep10",
		exist_ok=True,
		device="0",
        workers=0,

        # hsv_h=0.015,
        # hsv_s=0.5,
        # hsv_v=0.3,
        # augmentations=aug,
        # augment=True,
	)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--out", type=str, default="test")
    argv = parser.parse_args()
    train(
        argv.epoch,
    )

if __name__ == "__main__":
    main()
