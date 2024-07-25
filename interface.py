import cv2
from fer import FER
import threading
import tkinter as tk
from tkinter import messagebox


detector = FER(mtcnn=True)


camera_running = False

def start_camera():
    global camera_running
    camera_running = True
    threading.Thread(target=run_camera).start()

def stop_camera():
    global camera_running
    camera_running = False

def run_camera():
    global camera_running

    
    cap = cv2.VideoCapture(0)

    while camera_running:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.detect_emotions(frame)

        for result in results:
            bounding_box = result['box']
            emotions = result['emotions']
            x, y, w, h = bounding_box

            max_emotion = max(emotions, key=emotions.get)

            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(frame, max_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        cv2.imshow('Facial Expression Analysis', frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

   
    cap.release()
    cv2.destroyAllWindows()

# Setting up the GUI
root = tk.Tk()
root.title("Facial Expression Analysis")

start_button = tk.Button(root, text="Start Camera", command=start_camera)
start_button.pack(pady=20)

stop_button = tk.Button(root, text="Stop Camera", command=stop_camera)
stop_button.pack(pady=20)

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        stop_camera()
        root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()

