# Object Detection

## Usage

- ### training
	Please change the file path first.
	```python
	def train(
    	epochs: int = 10,
	):
    model = YOLO("yolo11s.pt")
    root = r"D:\NYCU\AI\hw4\dataset"    # <-- here
    data = {
		"nc": 1,
		"names": ["pig"],
		"train": os.path.join(root, "images\\train"), # <-- here
		"val": os.path.join(root, "images\\test"),  # <-- and here
	}
	```
	then run
	```ps
	python yolo.py
	```
- ### evaluate
  	```python
	# ============================================================
	# 1.Dataset設定
	# ============================================================
	DATA_ROOT = r"D:/NYCU/AI/hw4/dataset"       # ← 請改這裡(必要)
	DATA_ROOT = DATA_ROOT.replace("\\", "/")
  	```
	Then in the terminal,
	```ps
	python evaluate.py
	```

