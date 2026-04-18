import cv2
import os
import numpy as np
from PIL import Image

print("=" * 50)
print("ATTENDANCE SYSTEM DEBUGGER")
print("=" * 50)

# 1. Check if model exists
model_path = "TrainingImageLabel/Trainner.yml"
if os.path.exists(model_path):
    print("✅ Model file exists")
else:
    print("❌ Model file NOT found! Please save profile first")
    exit()

# 2. Load and test the model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)
print("✅ Model loaded successfully")

# 3. Load student data
student_file = "StudentDetails/StudentDetails.csv"
students = {}
if os.path.exists(student_file):
    import csv
    with open(student_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 5:
                try:
                    serial = int(row[0])
                    student_id = row[2]
                    name = row[4]
                    students[serial] = {'id': student_id, 'name': name}
                    print(f"  Loaded: Serial={serial}, ID={student_id}, Name={name}")
                except Exception as e:
                    print(f"  Error parsing row: {row} - {e}")
    print(f"✅ Loaded {len(students)} student(s)")
else:
    print("❌ Student file not found")
    exit()

# 4. Check Training Images
image_dir = "TrainingImage"
images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
print(f"\n📸 Training images found: {len(images)}")

if len(images) > 0:
    print("Sample images:")
    for img in images[:5]:
        print(f"  - {img}")
    
    # Test if images can be read
    test_img = images[0]
    img_path = os.path.join(image_dir, test_img)
    pil_img = Image.open(img_path).convert('L')
    img_np = np.array(pil_img, 'uint8')
    print(f"✅ Test image loaded: {img_np.shape}")

# 5. Test face detection on a sample image
print("\n🔍 Testing face detection...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Try to open camera
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("✅ Camera opened successfully")
    
    print("\n📷 Starting camera test...")
    print("Look at the camera. Press 's' to test recognition, 'q' to quit")
    
    test_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Get face ROI
            roi = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (200, 200))
            
            # Try to recognize
            try:
                serial, confidence = recognizer.predict(roi_resized)
                print(f"\n📊 Recognition Result:")
                print(f"  Predicted Serial: {serial}")
                print(f"  Confidence: {confidence} (lower is better)")
                
                if serial in students:
                    print(f"  ✅ Match found: {students[serial]['name']} (ID: {students[serial]['id']})")
                    cv2.putText(frame, f"{students[serial]['name']} ({confidence:.0f})", 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    print(f"  ❌ Serial {serial} not in student database")
                    cv2.putText(frame, f"Unknown (Serial:{serial})", 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                print(f"  {'='*40}")
                test_count += 1
                
            except Exception as e:
                print(f"Error during prediction: {e}")
                cv2.putText(frame, "Error", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.putText(frame, "Press 's' to test, 'q' to quit", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Debug - Face Recognition Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print("\n" + "="*50)
            print("Press 's' again to run another test")
            print("="*50)
    
    cap.release()
else:
    print("❌ Cannot open camera! Please check camera connection")

cv2.destroyAllWindows()
print("\nDebug session complete!")