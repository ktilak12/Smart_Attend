# clean_register.py
import cv2
import os
import csv
import numpy as np
from PIL import Image

# Clean old data
os.makedirs("TrainingImage", exist_ok=True)
os.makedirs("TrainingImageLabel", exist_ok=True)

# Clear old files
for f in os.listdir("TrainingImage"):
    os.remove(os.path.join("TrainingImage", f))

# Get student info
student_id = input("Enter Student ID (e.g., 101): ")
name = input("Enter Student Name: ")

# Get serial number
csv_file = "StudentDetails/StudentDetails.csv"
if os.path.exists(csv_file):
    with open(csv_file, 'r') as f:
        serial = len(list(csv.reader(f)))
else:
    serial = 1

print(f"\n📸 Taking 50 images for {name} (ID: {student_id}, Serial: {serial})")
print("Look at the camera...")

# Open camera
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
count = 0

while count < 50:
    ret, frame = cap.read()
    if not ret:
        continue
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        count += 1
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        
        # Save with correct format: Name.Serial.ID.Number.jpg
        filename = f"TrainingImage/{name}.{serial}.{student_id}.{count}.jpg"
        cv2.imwrite(filename, face)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"{count}/50", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"{name} ({student_id})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    cv2.imshow('Registering - Press ESC to stop early', frame)
    if cv2.waitKey(50) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n✅ Captured {count} images!")

# Save to CSV
with open(csv_file, 'a', newline='') as f:
    writer = csv.writer(f)
    if serial == 1:
        writer.writerow(['SERIAL NO.', '', 'ID', '', 'NAME'])
    writer.writerow([serial, '', student_id, '', name])

print(f"✅ Saved to CSV: Serial {serial}, ID {student_id}, Name {name}")

# Train the model
print("\n🔄 Training model...")
recognizer = cv2.face.LBPHFaceRecognizer_create()
faces = []
ids = []

for img_file in os.listdir("TrainingImage"):
    if img_file.endswith('.jpg'):
        img_path = os.path.join("TrainingImage", img_file)
        pil_img = Image.open(img_path).convert('L')
        img_np = np.array(pil_img, 'uint8')
        
        # Extract serial from filename (second part)
        parts = img_file.split('.')
        if len(parts) >= 2:
            serial_num = int(parts[1])  # This is the SERIAL, not the ID
            faces.append(img_np)
            ids.append(serial_num)

if faces:
    recognizer.train(faces, np.array(ids))
    recognizer.save("TrainingImageLabel/Trainner.yml")
    print(f"✅ Model trained with {len(faces)} images!")
    print(f"   Expected serial: {serial}")
else:
    print("❌ No faces found to train!")

print("\n" + "="*50)
print("REGISTRATION COMPLETE!")
print(f"Now when you take attendance, it should recognize Serial {serial}")
print("="*50)