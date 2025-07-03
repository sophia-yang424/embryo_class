import cv2
import time
import os
from datetime import datetime
from ultralytics import YOLO
import threading
import queue
import torch
from PIL import Image

# Initialize YOLO model (used for detection only)
model = YOLO('yolov8n.pt')

# Load a pretrained classification model (e.g., ResNet18)
import tensorflow as tf
classification_model = tf.keras.models.load_model('your_tensorflow_model.h5')

# Use custom labels for your TensorFlow model
imagenet_labels = ["bad", "good"]
imagenet_labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_labels = requests.get(imagenet_labels_url).text.strip().split("\n")

# Define transform for classifier input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def run_pretrained_classifier(cropped_img):
    image = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)).resize((224, 224))
    input_tensor = tf.keras.utils.img_to_array(image) / 255.0
    input_tensor = tf.expand_dims(input_tensor, 0)  # Add batch dimension
    output = classification_model.predict(input_tensor)
    predicted = tf.argmax(output[0]).numpy()
    return predicted

# Open webcam
cap = cv2.VideoCapture(0)

PERSISTENCE_TIME = 5.0
CONFIDENCE_THRESHOLD = 0.5
SAVE_DIRECTORY = "C:/Users/sophi/OneDrive/Desktop/embryo_directory"
os.makedirs(SAVE_DIRECTORY, exist_ok=True)

detection_queue = queue.Queue()
tracked_object = None  # initialize the tracked object

def is_inside_roi(box, frame_width, frame_height):
    roi_x1 = int(frame_width * 0.3)
    roi_y1 = int(frame_height * 0.3)
    roi_x2 = int(frame_width * 0.7)
    roi_y2 = int(frame_height * 0.7)
    x1, y1, x2, y2 = box
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return roi_x1 <= center_x <= roi_x2 and roi_y1 <= center_y <= roi_y2

def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    return intersection / union if union > 0.5 else 0.0

def save_detection(frame, box):
    x1, y1, x2, y2 = box
    cropped = frame[y1:y2, x1:x2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"good_{timestamp}.jpg"
    filepath = os.path.join(SAVE_DIRECTORY, filename)
    cv2.imwrite(filepath, cropped)
    print(f"[SAVED] good saved as {filename}")
    return filepath

def send_signal():
    print(f"[SIGNAL] Good embryo detected!")

def classify_object(frame, box):
    x1, y1, x2, y2 = box
    cropped = frame[y1:y2, x1:x2]
    result = run_pretrained_classifier(cropped)
    label = imagenet_labels[result]
    print(f"[CLASSIFIED] Class index: {result}, Label: {label}")
    if label.lower() in ["person", "tench", "goldfish", "embryo"]:
        save_detection(frame, box)
        send_signal()
    else:
        print(f"[SIGNAL] Bad embryo detected. Label: {label}")

def classification_worker():
    while True:
        try:
            task = detection_queue.get(timeout=1.0)
            if task is None:
                break
            frame, box = task
            classify_object(frame, box)
            detection_queue.task_done()
        except queue.Empty:
            continue

classification_thread = threading.Thread(target=classification_worker, daemon=True)
classification_thread.start()

print("Starting detection... Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_time = time.time()
        frame_height, frame_width = frame.shape[:2]

        cv2.rectangle(frame, (int(frame_width * 0.3), int(frame_height * 0.3)),
                      (int(frame_width * 0.7), int(frame_height * 0.7)), (255, 0, 255), 2)

        results = model(frame)[0]
        detection_found = False

        for box in results.boxes:
            score = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if score < CONFIDENCE_THRESHOLD:
                continue
            if not is_inside_roi((x1, y1, x2, y2), frame_width, frame_height):
                continue

            detection_found = True
            current_box = (x1, y1, x2, y2)

            if tracked_object is None:
                tracked_object = {
                    'box': current_box,
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'classified': False
                }
            else:
                iou = calculate_iou(current_box, tracked_object['box'])
                if iou > 0.3:
                    tracked_object['box'] = current_box
                    tracked_object['last_seen'] = current_time
                    elapsed = current_time - tracked_object['first_seen']
                    if elapsed >= PERSISTENCE_TIME and not tracked_object['classified']:
                        detection_queue.put((frame.copy(), current_box))
                        tracked_object['classified'] = True
                    color = (0, 255, 0) if tracked_object['classified'] else (0, 255, 255)
                    status = "CLASSIFIED" if tracked_object['classified'] else f"WAIT ({PERSISTENCE_TIME - elapsed:.1f}s)"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                else:
                    tracked_object = {
                        'box': current_box,
                        'first_seen': current_time,
                        'last_seen': current_time,
                        'classified': False
                    }
            break

        if not detection_found:
            tracked_object = None

        cv2.imshow("Embryo Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("Shutting down...")
    detection_queue.put(None)
    cap.release()
    cv2.destroyAllWindows()
    classification_thread.join(timeout=2.0)
    print("Shutdown complete.")
