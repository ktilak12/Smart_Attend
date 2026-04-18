# check_images.py
import os

image_dir = "TrainingImage"
for f in os.listdir(image_dir)[:10]:
    if f.endswith('.jpg'):
        parts = f.split('.')
        print(f"File: {f}")
        print(f"  Parts: {parts}")
        if len(parts) >= 3:
            print(f"  Extracted ID: {parts[2]}")
        print("-" * 30)