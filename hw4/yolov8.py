from ultralytics.models.yolo.model import YOLO
import rich, rich.progress

class RichProgressCallback:
    def __init__(self):
        self.progress = rich.progress.Progress(
			rich.progress.TextColumn("[bold blue] Epoch {task.fields[epochs]}/{task.fields[total_epochs]}"),
		)

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
