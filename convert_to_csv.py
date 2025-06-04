import os
import csv
from ultralytics import YOLO
from PIL import Image

# Path to the directory containing test images
image_dir = './data/val/images'
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Load the YOLOv8 model
model = YOLO('your_model.pt')  # or path to your custom .pt model

# Output CSV file
output_csv = 'submission.csv'

with open(output_csv, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'predictions'])  # CSV header

    for image_name in image_files:
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path)
        width, height = image.size

        # Run inference
        results = model(image_path)[0]

        predictions = []
        for box in results.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())

            # xywh in pixels (center x, center y, width, height)
            xywh = box.xywh[0]
            x = float(xywh[0]) / width
            y = float(xywh[1]) / height
            w = float(xywh[2]) / width
            h = float(xywh[3]) / height

            predictions.append([cls_id, x, y, w, h, conf])

        writer.writerow([image_name, predictions])
