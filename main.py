import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mess
import tkinter.simpledialog as tsd
import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
import threading
import requests  # pip install requests

# ── ESP CONFIG: Set your ESP's IP address here ─────────────────────────────
ESP_IP = "192.168.4.1"   # <-- Replace with the IP shown in ESP Serial Monitor
ESP_URL = f"http://{ESP_IP}/attendance"
ESP_ENABLED = True         # Set False to run without ESP hardware
# ───────────────────────────────────────────────────────────────────────────

for folder in ['TrainingImage', 'TrainingImageLabel', 'StudentDetails', 'Attendance']:
    os.makedirs(folder, exist_ok=True)

window = None
message1 = None
message = None
txt = None
txt2 = None
tv = None
clock = None

def assure_path_exists(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def tick():
    time_string = time.strftime('%H:%M:%S')
    if clock:
        clock.config(text=time_string)
    window.after(200, tick)

def clear():
    if txt and txt2:
        txt.delete(0, 'end')
        txt2.delete(0, 'end')
        if message1:
            message1.configure(text="1)Take Images  >>>  2)Save Profile")

def trigger_esp_led(student_name):
    """Send HTTP GET to ESP in background thread — won't freeze the GUI."""
    if not ESP_ENABLED:
        return
    def _send():
        try:
            resp = requests.get(ESP_URL, params={"name": student_name}, timeout=5)
            print(f"[ESP] Response: {resp.text}")
        except requests.exceptions.ConnectionError:
            print(f"[ESP] Could not connect to {ESP_IP}. Check IP and WiFi.")
        except requests.exceptions.Timeout:
            print("[ESP] Request timed out.")
        except Exception as e:
            print(f"[ESP] Error: {e}")
    threading.Thread(target=_send, daemon=True).start()

def TakeImages():
    if not txt or not txt2:
        return
    Id = txt.get().strip()
    name = txt2.get().strip()
    if not Id or not name:
        mess._show(title='Error', message='Please enter both ID and Name')
        return
    if not Id.isdigit():
        mess._show(title='Error', message='ID must be a number!')
        return

    csv_file = "StudentDetails/StudentDetails.csv"
    columns = ['SERIAL NO.', '', 'ID', '', 'NAME']
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
        serial = 1
    else:
        with open(csv_file, 'r') as f:
            rows = [l for l in csv.reader(f) if l]
            serial = max(1, len(rows) - 1)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        mess._show(title='Error', message='Cannot open camera!')
        return

    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    if detector.empty():
        mess._show(title='Error', message='haarcascade_frontalface_default.xml not found!')
        cam.release()
        return

    sampleNum = 0
    total_images = 50
    if message1:
        message1.configure(text=f"Taking {total_images} images... Look at camera")

    while sampleNum < total_images:
        ret, img = cam.read()
        if not ret:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            sampleNum += 1
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_resized = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            cv2.imwrite(f"TrainingImage/{name}.{serial}.{Id}.{sampleNum}.jpg", face_resized)
            cv2.putText(img, f"{sampleNum}/{total_images}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Taking Images - Press ESC to stop', img)
        if cv2.waitKey(50) & 0xFF == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

    if sampleNum > 0:
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([serial, '', Id, '', name])
        if message1:
            message1.configure(text=f"Captured {sampleNum} images for ID: {Id}")
        if message:
            message.configure(text="Now click 'Save Profile' to train the model")
        mess._show(title='Success', message=f'Captured {sampleNum} images!\nClick Save Profile to continue.')
    else:
        if message1:
            message1.configure(text="No faces detected! Try better lighting.")

def psw():
    assure_path_exists("TrainingImageLabel/")
    password_file = "TrainingImageLabel/psd.txt"
    if not os.path.exists(password_file):
        new_pas = tsd.askstring('Password Setup', 'Create a new password:', show='*')
        if new_pas:
            with open(password_file, 'w') as f:
                f.write(new_pas)
            mess._show(title='Success', message='Password created! Click Save Profile again.')
        return
    with open(password_file, 'r') as f:
        key = f.read()
    password = tsd.askstring('Password', 'Enter password to save profile:', show='*')
    if password == key:
        TrainImages()
    elif password:
        mess._show(title='Wrong Password', message='Incorrect password!')

def TrainImages():
    image_dir = "TrainingImage"
    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    if len(images) == 0:
        mess._show(title='Error', message='No images found! Please take images first.')
        return
    if message1:
        message1.configure(text="Training model... Please wait")
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        faces = []
        ids = []
        for img_file in images:
            pil_img = Image.open(os.path.join(image_dir, img_file)).convert('L')
            img_np = cv2.resize(np.array(pil_img, 'uint8'), (200, 200))
            parts = img_file.split('.')
            if len(parts) >= 4:
                try:
                    student_id = int(parts[2])
                    faces.append(img_np)
                    ids.append(student_id)
                except ValueError:
                    pass
        if len(faces) > 0:
            recognizer.train(faces, np.array(ids))
            recognizer.save("TrainingImageLabel/Trainner.yml")
            if message1:
                message1.configure(text=f"Profile saved! Trained with {len(faces)} images.")
            if message:
                message.configure(text=f'Total Registrations: {len(set(ids))}')
            mess._show(title='Success', message=f'Profile saved!\nTrained with {len(faces)} images.')
        else:
            mess._show(title='Error', message='No valid faces found!')
    except Exception as e:
        mess._show(title='Error', message=f'Training failed: {str(e)}')

def TrackImages():
    if not os.path.exists("TrainingImageLabel/Trainner.yml"):
        mess._show(title='Error', message='Please save profile first!')
        return
    if tv:
        for item in tv.get_children():
            tv.delete(item)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel/Trainner.yml")
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        mess._show(title='Error', message='haarcascade_frontalface_default.xml not found!')
        return
    if not os.path.exists("StudentDetails/StudentDetails.csv"):
        mess._show(title='Error', message='Student details missing!')
        return

    # Key by student ID (int) — critical fix
    students = {}
    with open("StudentDetails/StudentDetails.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 5:
                try:
                    sid = int(row[2])
                    students[sid] = {'id': str(sid), 'name': row[4]}
                except (ValueError, IndexError):
                    pass

    if len(students) == 0:
        mess._show(title='Error', message='No student data found!')
        return

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        mess._show(title='Error', message='Cannot open camera!')
        return

    marked_students = set()
    today = datetime.datetime.now().strftime('%d-%m-%Y')
    att_file = f"Attendance/Attendance_{today}.csv"
    if not os.path.exists(att_file):
        with open(att_file, 'w', newline='') as f:
            csv.writer(f).writerow(['ID', 'Name', 'Date', 'Time'])

    if message1:
        message1.configure(text="Taking Attendance... Press 'q' to quit")

    while True:
        ret, im = cam.read()
        if not ret:
            continue
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi_resized = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            try:
                predicted_id, confidence = recognizer.predict(roi_resized)
                if confidence < 100 and predicted_id in students:
                    name = students[predicted_id]['name']
                    student_id = students[predicted_id]['id']
                    cv2.putText(im, f"{name} ({student_id}) [{int(confidence)}]",
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    if predicted_id not in marked_students:
                        marked_students.add(predicted_id)
                        now = datetime.datetime.now()
                        time_stamp = now.strftime('%H:%M:%S')
                        with open(att_file, 'a', newline='') as f:
                            csv.writer(f).writerow([student_id, name, today, time_stamp])
                        if tv:
                            tv.insert('', 0, text=student_id, values=(name, today, time_stamp))
                        if message1:
                            message1.configure(text=f"Marked: {name}")

                        # ── TRIGGER ESP LED ──────────────────────────────
                        trigger_esp_led(name)
                        # ────────────────────────────────────────────────
                else:
                    cv2.putText(im, f"Unknown [{int(confidence)}]",
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            except Exception:
                cv2.putText(im, "Error", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Attendance - Press Q to quit', im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    if len(marked_students) > 0:
        if message1:
            message1.configure(text=f"Attendance taken for {len(marked_students)} students!")
        mess._show(title='Success', message=f'Attendance recorded for {len(marked_students)} student(s)!')
    else:
        if message1:
            message1.configure(text="No students recognized!")
        mess._show(title='Warning', message='No faces recognized.\nCheck lighting and re-register if needed.')


# ── GUI ────────────────────────────────────────────────────────────────────────
window = tk.Tk()
window.geometry("1280x720")
window.title("Attendance System")
window.configure(background='#262523')

frame1 = tk.Frame(window, bg="#00aeff")
frame1.place(relx=0.11, rely=0.17, relwidth=0.39, relheight=0.80)
frame2 = tk.Frame(window, bg="#00aeff")
frame2.place(relx=0.51, rely=0.17, relwidth=0.38, relheight=0.80)

tk.Label(window, text="Face Recognition Based Attendance System", fg="white",
         bg="#262523", width=55, height=1, font=('times', 29, ' bold ')).place(x=10, y=10)

frame3 = tk.Frame(window, bg="#c4c6ce")
frame3.place(relx=0.52, rely=0.09, relwidth=0.09, relheight=0.07)
frame4 = tk.Frame(window, bg="#c4c6ce")
frame4.place(relx=0.36, rely=0.09, relwidth=0.16, relheight=0.07)

tk.Label(frame4, text=datetime.datetime.now().strftime('%d-%B-%Y'), fg="orange",
         bg="#262523", width=55, height=1, font=('times', 22, ' bold ')).pack(fill='both', expand=1)

clock = tk.Label(frame3, fg="orange", bg="#262523", width=55, height=1, font=('times', 22, ' bold '))
clock.pack(fill='both', expand=1)
tick()

tk.Label(frame2, text="For New Registrations", fg="black", bg="#3ece48",
         font=('times', 17, ' bold ')).grid(row=0, column=0)
tk.Label(frame1, text="For Already Registered", fg="black", bg="#3ece48",
         font=('times', 17, ' bold ')).place(x=0, y=0)

tk.Label(frame2, text="Enter ID", width=20, height=1, fg="black", bg="#00aeff",
         font=('times', 17, ' bold ')).place(x=80, y=55)
txt = tk.Entry(frame2, width=32, fg="black", font=('times', 15, ' bold '))
txt.place(x=30, y=88)

tk.Label(frame2, text="Enter Name", width=20, fg="black", bg="#00aeff",
         font=('times', 17, ' bold ')).place(x=80, y=140)
txt2 = tk.Entry(frame2, width=32, fg="black", font=('times', 15, ' bold '))
txt2.place(x=30, y=173)

message1 = tk.Label(frame2, text="1)Take Images  >>>  2)Save Profile", bg="#00aeff",
                    fg="black", width=39, height=1, font=('times', 15, ' bold '))
message1.place(x=7, y=230)

message = tk.Label(frame2, text="", bg="#00aeff", fg="black", width=39, height=1,
                   font=('times', 16, ' bold '))
message.place(x=7, y=450)

tk.Label(frame1, text="Attendance", width=20, fg="black", bg="#00aeff",
         height=1, font=('times', 17, ' bold ')).place(x=100, y=115)

tv = ttk.Treeview(frame1, height=13, columns=('name', 'date', 'time'))
tv.column('#0', width=82); tv.column('name', width=130)
tv.column('date', width=133); tv.column('time', width=133)
tv.grid(row=2, column=0, padx=(0,0), pady=(150,0), columnspan=4)
tv.heading('#0', text='ID'); tv.heading('name', text='NAME')
tv.heading('date', text='DATE'); tv.heading('time', text='TIME')

scroll = ttk.Scrollbar(frame1, orient='vertical', command=tv.yview)
scroll.grid(row=2, column=4, padx=(0,100), pady=(150,0), sticky='ns')
tv.configure(yscrollcommand=scroll.set)

tk.Button(frame2, text="Clear", command=clear, fg="black", bg="#ea2a2a",
          width=11, font=('times', 11, ' bold ')).place(x=335, y=86)
tk.Button(frame2, text="Take Images", command=TakeImages, fg="white", bg="blue",
          width=34, height=1, font=('times', 15, ' bold ')).place(x=30, y=300)
tk.Button(frame2, text="Save Profile", command=psw, fg="white", bg="blue",
          width=34, height=1, font=('times', 15, ' bold ')).place(x=30, y=380)
tk.Button(frame1, text="Take Attendance", command=TrackImages, fg="black", bg="yellow",
          width=35, height=1, font=('times', 15, ' bold ')).place(x=30, y=50)
tk.Button(frame1, text="Quit", command=window.destroy, fg="black", bg="red",
          width=35, height=1, font=('times', 15, ' bold ')).place(x=30, y=450)

res = 0
if os.path.exists("StudentDetails/StudentDetails.csv"):
    with open("StudentDetails/StudentDetails.csv", 'r') as f:
        res = max(0, len([l for l in csv.reader(f) if l]) - 1)
message.configure(text=f'Total Registrations till now : {res}')

window.mainloop()
