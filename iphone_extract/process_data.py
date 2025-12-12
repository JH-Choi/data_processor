import cv2
import os

video_path = "/mnt/hdd/code/Dongki_project/Genesis3DGS/data/IMG_8722.MOV"
output_dir = "./IMG_8722"
os.makedirs(output_dir, exist_ok=True)

assert os.path.exists(video_path)
cap = cv2.VideoCapture(video_path)

sharpness_threshold = 100.0  # adjust depending on the dataset
frame_id = 0
saved_id = -1
sampling = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    if variance > sharpness_threshold:
        saved_id += 1
        if saved_id % sampling != 0:
            continue
    
        cv2.imwrite(f"{output_dir}/{saved_id:05d}.jpg", frame)

    frame_id += 1

cap.release()

