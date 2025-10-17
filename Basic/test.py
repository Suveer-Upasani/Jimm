import cv2
import numpy as np
import requests

# Phone camera MJPEG URL
phone_url = "http://192.168.0.124:8080/video"

# Laptop webcam
cap_laptop = cv2.VideoCapture(0)

# Phone stream using requests
phone_stream = requests.get(phone_url, stream=True)
bytes_data = b''

while True:
    # ---------------- Phone camera ----------------
    for chunk in phone_stream.iter_content(chunk_size=1024):
        bytes_data += chunk
        a = bytes_data.find(b'\xff\xd8')  # JPEG start
        b = bytes_data.find(b'\xff\xd9')  # JPEG end
        if a != -1 and b != -1:
            jpg = bytes_data[a:b+2]
            bytes_data = bytes_data[b+2:]
            frame_phone = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            break  # Break to process laptop frame

    # ---------------- Laptop camera ----------------
    ret, frame_laptop = cap_laptop.read()
    if not ret:
        print("Failed to grab laptop frame")
        break

    # Resize both frames to same height
    height = 480
    frame_phone = cv2.resize(frame_phone, (int(frame_phone.shape[1]*height/frame_phone.shape[0]), height))
    frame_laptop = cv2.resize(frame_laptop, (int(frame_laptop.shape[1]*height/frame_laptop.shape[0]), height))

    # Concatenate frames horizontally
    combined_frame = np.hstack((frame_laptop, frame_phone))

    cv2.imshow("Laptop + Phone Camera", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_laptop.release()
cv2.destroyAllWindows()
