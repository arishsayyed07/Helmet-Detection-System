import cv2
import math
import cvzone
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load YOLO model
model = YOLO("Weights/best.pt")

# Initialize video capture
video_path = "Media/test5.mp4"
cap = cv2.VideoCapture(video_path)

# Skip frames for smoother playback
frame_skip = 2
frame_count = 0

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Resize frame for faster processing
    img_resized = cv2.resize(img, (640, 360))

    # Run YOLO inference
    confidence_threshold = 0.5
    results = model(img_resized, stream=True, conf=confidence_threshold)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            # Only draw boxes with sufficient confidence
            if conf > confidence_threshold:
                # Scale bounding box to original size
                x1 = int(x1 * img.shape[1] / img_resized.shape[1])
                y1 = int(y1 * img.shape[0] / img_resized.shape[0])
                x2 = int(x2 * img.shape[1] / img_resized.shape[1])
                y2 = int(y2 * img.shape[0] / img_resized.shape[0])
                w, h = x2 - x1, y2 - y1

                # Draw bounding box
                cvzone.cornerRect(img, (x1, y1, w, h), colorR=(0, 255, 0), l=30, t=3)

                # Define class names
                classNames = ["Helmet", "No Helmet"]  # Swap if needed

                label = f"{classNames[cls]} {conf}"
                cvzone.putTextRect(img, label, (max(0, x1), max(35, y1)), scale=1, thickness=1, colorR=(0, 0, 255))

    # **Replace cv2.imshow() with Matplotlib**
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    plt.axis("off")
    plt.show(block=False)
    plt.pause(0.01)  # Pause to display frame
    plt.clf()  # Clear figure for next frame

    # Exit on 'q' key press
    if plt.waitforbuttonpress(timeout=0.01) and plt.get_current_fig_manager().window:
        break

cap.release()
plt.close()
