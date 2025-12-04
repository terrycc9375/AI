from ultralytics.models.yolo.model import YOLO
from ultralytics.engine.trainer import BaseTrainer
import rich, rich.progress
import time

class RichProgressCallback:
    def __init__(self):
        self.progress = rich.progress.Progress(
			rich.progress.TextColumn("[bold blue] Epoch {task.fields[epochs]}/{task.fields[total_epochs]}"),
            rich.progress.BarColumn(bar_width=20),
            rich.progress.MofNCompleteColumn(),
            rich.progress.TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            rich.progress.TextColumn("•"),
            rich.progress.TextColumn("{task.fields[time_info]}"),
            transient=False
		)
        
        self.task_id = None
        self.start_time = None
        
    def on_train_start(self, trainer: BaseTrainer):
        self.task_id = self.progress.add_task(
            description=f"[green]Training",
            epoch=0,
            total=trainer.epochs
        )
        self.progress.start()

    def on_train_end(self, trainer: BaseTrainer):
        self.progress.update(self.task_id, completed=trainer.epochs)
        self.progress.stop()

def train():
    model = YOLO("yolo8n.pt")
    root = r"D:/NYCU/AI/hw4/dataset/images"
    data = {
		"nc": 1,
		"name": ["pig"],
		"train": f"{root}/train",
		"val": f"{root}/test",
		"test": f"{root}/test"
	}
    training_logs = model.train(
		data=data,
		epoch=4,
		batch=16,
		imgsz=320,
		project="YOLOv8",
		name="test",
		exist_ok=True,
		device="1"
	)

def main():
    pass

if __name__ == "__main__":
    main()
