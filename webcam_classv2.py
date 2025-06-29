import cv2
from ultralytics import YOLO

#in this version, i made it so that if objects are not in the roi, we dont classify.
#to avoid it defaulting to doing a bad guess when theres notinng relevant, which wastes computing power,
#i made it so it only continues if the guess is above >0.5 confidence
model = YOLO('yolov8n.pt')

# open webcam
cap = cv2.VideoCapture(0)

def is_inside_roi(box, frame_width, frame_height):
    roi_x1 = int(frame_width * 0.3)
    roi_y1 = int(frame_height * 0.3)
    roi_x2 = int(frame_width * 0.7)
    roi_y2 = int(frame_height * 0.7)

    x1, y1, x2, y2 = box
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    return roi_x1 <= center_x <= roi_x2 and roi_y1 <= center_y <= roi_y2

while True:
    ret, frame = cap.read()
    #if cam has error we quit (ret is bool, frame is frame object of whole screen)
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]

    # defining the fixed roi box (where we look for stuff)
    cv2.rectangle(frame, 
                  (int(frame_width * 0.3), int(frame_height * 0.3)), 
                  (int(frame_width * 0.7), int(frame_height * 0.7)), 
                  (255, 0, 255), 2)

    
#run the yolo detection on our frame object
    results = model(frame)[0]

    # loop through detected objects
    for box in results.boxes:
        cls_id = int(box.cls[0])
        score = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[cls_id]
        if score < 0.5:
            continue
        if not is_inside_roi((x1, y1, x2, y2), frame_width, frame_height):
            continue  # skip anything outside

        in_roi = is_inside_roi((x1, y1, x2, y2), frame_width, frame_height)
        color = (0, 255, 0) if in_roi else (0, 0, 255)

        if in_roi:
            print(f"[DETECTED IN ROI] {label} with {score:.2f} confidence.")

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{label} {score:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # display the result
    cv2.imshow("YOLO Detection", frame)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#quit
cap.release()
cv2.destroyAllWindows()
