"""
draw bounding boxes for one labeled image
"""

import cv2, argparse, os, typing

def draw(img, label_line: str):
    h, w = img.shape[:2] # type: ignore
    parts = label_line.strip().split()
    class_id = parts[0]
    cx, cy, bw, bh = map(float, parts[1:])
    
    x1 = int((cx - bw/2) * w)
    y1 = int((cy - bh/2) * h)
    x2 = int((cx + bw/2) * w)
    y2 = int((cy + bh/2) * h)
    
    cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2) # type: ignore
    center_x_pixel = int(cx * w)
    center_y_pixel = int(cy * h)
    cv2.circle(img, (center_x_pixel, center_y_pixel), 8, (0, 0, 255), -1) #type: ignore
    return img

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", type=str, required=True)
	argv = parser.parse_args()

	if len(argv.i) != 8:
		raise ValueError("Unsupported include file")

	path = os.path.join(".\\dataset\\images\\train\\", argv.i + ".jpg")
	label = os.path.join(".\\dataset\\labels\\train\\", argv.i + ".txt")

	img = cv2.imread(path)
	with open(label, 'r') as f:
		content = f.readlines()
		for line in content:
			img = draw(img, line) if len(line) > 0 else img
    
	cv2.imshow("YOLO label visualization", img) # type: ignore
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
