from ultralytics.models.yolo.model import YOLO
from ultralytics.engine.trainer import BaseTrainer
import rich, rich.progress
import time
import argparse

console = rich.console.Console()
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
        self.start_time = float()
        
    def on_train_start(self, trainer: BaseTrainer):
        self.task_id = self.progress.add_task(
            description=f"[green]Training",
            epoch=0,
            total=trainer.epochs
        )
        self.progress.start()
        self.start_time = time.time()

    def on_fit_epoch_end(self, trainer: BaseTrainer):
        elapsed = time.time() - self.start_time
        current_epoch = trainer.epoch
        self.progress.update(
            self.task_id, # type: ignore
            total=trainer.epochs,
            completed=current_epoch + 1,
            epoch=current_epoch + 1,
            advance=1
        ) 

    def on_train_end(self, trainer: BaseTrainer):
        self.progress.update(self.task_id, completed=trainer.epochs) # type: ignore
        self.progress.stop()
        console.print(f"\n[bold magenta]Training complete.\n[bold #6edba1]")

def train(
    epochs: int = 4,
):
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
		epoch=epochs,
		batch=16,
		imgsz=320,
		project="YOLOv8",
		name="test",
		exist_ok=True,
		device="1"
	)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--out", type=str, default="test")
    argv = parser.parse_args()
    train(
        argv.epoch,
    )

if __name__ == "__main__":
    main()
