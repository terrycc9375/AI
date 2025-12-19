from ultralytics.models.yolo.model import YOLO
from ultralytics.engine.trainer import BaseTrainer
import rich, rich.progress
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

console = rich.console.Console()
class RichProgressCallback:
    def __init__(self):
        pass
        self.progress = rich.progress.Progress(
			rich.progress.TextColumn("[bold blue] Epoch {task.fields[epoch]}/{task.fields[total]}"),
            rich.progress.BarColumn(bar_width=20),
            rich.progress.MofNCompleteColumn(),
            rich.progress.TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            # rich.progress.TextColumn("•"),
            # rich.progress.TextColumn("{task.fields[time_info]}"),
            transient=False
		)
        
        self.start_time = float()
        self.batch_task = None
        self.epoch_task = None
        
    def on_train_start(self, trainer: BaseTrainer):
        pass
        self.epoch_task = self.progress.add_task(
            description=f"[green]Training",
            epoch=0,
            total=trainer.epochs
        )
        self.progress.start()
        self.start_time = time.time()
        
    def on_train_epoch_start(self, trainer: BaseTrainer):
        pass
        self.epoch_start_time = time.time()
        batches = trainer.train_loader.__len__()
        if self.batch_task is not None:
            self.progress.remove_task(self.batch_task)
            
        self.batch_task = self.progress.add_task(
            description="[yellow]  batch",
            total=batches,
            completed=0,
            visible=True,
        )
        self.batch_start_time = time.time()
        self.progress.update(
            self.epoch_task, # type: ignore
            advance=1,
            epoch=trainer.epoch + 1,
            completed=trainer.epoch + 1,
        )
        
    def on_train_batch_end(self, trainer: BaseTrainer):
        pass
        if self.batch_task is None:
            return
        batch_idx = trainer.batch_idx + 1 # type: ignore
        batches = trainer.train_loader.__len__()
        elapsed = time.time() - self.batch_start_time
        speed = batch_idx / elapsed if elapsed > 0 else 0.0
        self.progress.update(
            self.batch_task,
            advance=1,
            completed=batch_idx,
            description=f"[yellow]  batch {batch_idx}/{batches}, {speed:.2f} it/s"
        )

    def on_fit_epoch_end(self, trainer: BaseTrainer):
        pass
        elapsed = time.time() - self.start_time
        self.progress.update(
            self.epoch_task, # type: ignore
            completed=trainer.epoch + 1,
            epoch=trainer.epoch + 1,
            advance=1
        )
        if self.batch_task is not None:
            self.progress.update(
                self.batch_task,
                completed=trainer.train_loader.__len__()
            )

    def on_train_end(self, trainer: BaseTrainer):
        pass
        if self.batch_task is not None:
            self.progress.remove_task(self.batch_task)
        self.progress.update(self.epoch_task, completed=trainer.epochs) # type: ignore
        self.progress.stop()
        console.print(f"\n[bold magenta]Training complete.\n[bold #6edba1]")

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
		batch=8,
		imgsz=1280,
		project="YOLO11",
		name="ep10",
		exist_ok=True,
		device="0",
        workers=0,

        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        augmentations=aug,
        augment=True,
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
