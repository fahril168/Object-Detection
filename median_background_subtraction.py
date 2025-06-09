import cv2
import numpy as np
import requests
import time
import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": message
    }
    requests.post(url, data=data)

def send_telegram_photo(photo_path, caption=""):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    with open(photo_path, 'rb') as photo:
        data = {
            "chat_id": CHAT_ID,
            "caption": caption
        }
        files = {
            "photo": photo
        }
        requests.post(url, data=data, files=files)

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Gagal membuka kamera.")
    exit()

print("Tekan tombol 'b' untuk set background...")

background_set = False
gray_background = None
frames = []

notif_pending = False
first_detect_time = 0
last_notif_time = 0
notif_interval = 10  # detik

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(cv2.flip(frame, 1), (640, 480))  # Flip horizontal
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('b'):
        print("Mengambil background...")
        frames = []
        for _ in range(30):
            ret2, f = cap.read()
            if not ret2:
                continue
            f = cv2.resize(cv2.flip(f, 1), (640, 480))
            frames.append(f)
        frame_median = np.median(frames, axis=0).astype(np.uint8)
        gray_background = cv2.cvtColor(frame_median, cv2.COLOR_BGR2GRAY)
        background_set = True
        print("Background berhasil diset!")

    object_count = 0
    if background_set:
        diff = cv2.absdiff(gray_frame, gray_background)
        blur = cv2.GaussianBlur(diff, (7, 7), 0)
        _, thresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_objects = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                valid_objects.append((x, y, w, h))
        object_count = len(valid_objects)

        valid_objects.sort(key=lambda b: b[0])

        for i, (x, y, w, h) in enumerate(valid_objects):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, str(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        now = time.time()

        if object_count > 0:
            if not notif_pending:
                first_detect_time = now
                notif_pending = True
            elif now - first_detect_time >= 3 and now - last_notif_time >= notif_interval:
                photo_path = "deteksi.jpg"
                cv2.imwrite(photo_path, frame)
                send_telegram_message(f"Deteksi objek aktif: {object_count} objek terdeteksi.")
                send_telegram_photo(photo_path, caption="Gambar objek yang terdeteksi.")
                last_notif_time = now
                notif_pending = False
        else:
            notif_pending = False

        cv2.imshow("Threshold", thresh)

    cv2.putText(frame, f"Jumlah Objek: {object_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Deteksi Gerakan - Kamera (Flip)", frame)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
