import cv2
import numpy as np

def get_imgs_from_video(video_path, n=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Неверный путь")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    images = []
    for i in range(frame_count):
        ret, frame = cap.read()
        
        if i % n == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(img)
    images = np.array(images)
    return images