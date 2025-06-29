import cv2
# cv2 is OpenCV= library for image/video processing
from ultralytics import YOLO

# loading the pretrained YOLOv8 model (YOLO is trained on COCO)
model = YOLO("yolov8n.pt")  

# this function determines if we have something inside the box we are foxusing on: (returns bool)
#If yes → take a snapshot or classify.
#If no → ignore the frame.
#idea: only classify if an object enters this square
def is_inside_roi(box, frame_width, frame_height):
    # Define a centered square ROI
    roi_x1 = int(frame_width * 0.3)
    roi_y1 = int(frame_height * 0.3)
    roi_x2 = int(frame_width * 0.7)
    roi_y2 = int(frame_height * 0.7)
    #i defined the box here to be the middle 40% of the screen, but we can modify that
    # Get object's box center
    x1, y1, x2, y2 = box
    #(assume box arg we passs in is 4 coords)
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    #this way our thing doesnt get thrown off by edges or corners that are in the box, and instead only objects
    #  who are centered in this region
    
    return roi_x1 < center_x < roi_x2 and roi_y1 < center_y < roi_y2
#we use that function to determine if we want to trigger classification

# start webcam
cap = cv2.VideoCapture(0)
#starts capturing video from my default webcam (0 is the default device index, for externnal cams use 1,2 etc).
while True:
    #an infinite loop until we press q/use interface to tell it to quit

    #cap is the variable we defined above as the camera object
    #.read() is a function of the cv2 video object
    #read() returns 2 args: arg 1.) a bool (True if a frame was successfully captured, False if there was an error i.e camera disconnected etc)
    #arg 2.) frame aka the image frame (like the dims etc)
    ret, frame = cap.read()
    #frame = entire screen view from the camera at that moment.
    #frame = what we pass to YOLO for it to THEN detect objects

    if not ret:
        #so if there was an error in getting a frame, we quit
        break

    frame_height, frame_width = frame.shape[:2]
    #we get the first 2 fields of the frame object read.() returned
    #aka hieght and width of the frame

    # run the YOLO model on the frame (we defined our YOLO object as named "model")
    #take only the top/most likely prediction, since yolo gives a list of many possible predictions
    #if we want all of the predictions, can modify this
    
    results = model(frame)[0]
    #technically since for us, we pass one frame at a time, so doing the [0] is redundant
    #  its like a list with one object, but to get the object we still need to index to get first val
    #makes it scalable in case we do more frames at once
    

  #magenta box repreent our ROI (the box we use to determine whether to trigger classification)
  # this box is fixed, its where we are wathcing for objects
    cv2.rectangle(frame, (int(frame_width * 0.3), int(frame_height * 0.3)),
                  (int(frame_width * 0.7), int(frame_height * 0.7)), (255, 0, 255), 2)

    for box in results.boxes:
        #Iterates through all objects detected by YOLO in the current frame.
        cls_id = int(box.cls[0])
        score = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        #cls_id: the class number of the detected object (e.g. 0 = person, 1 = car)
        #score: confidence level of the detection (0 to 1) (ex 0.92 = 92% confident)
        #x1, y1, x2, y2: top-left and bottom-right corners of the bounding box

        label = model.names[cls_id]
        color = (0, 255, 0) if is_inside_roi((x1, y1, x2, y2), frame_width, frame_height) else (0, 0, 255)
#label: what we think it is (for YOLO, itll be stuff it was pretrained on like lamp, person etc)
#color = red if it is not in ROI, green if it is in ROI

        # only if green/if it was in ROI do we print this
        if color == (0, 255, 0):
            print(f"[DETECTED] {label} with {score:.2f} confidence inside region.")
        #printed in terminal, not displayed on screen


        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{label} {score:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
#Draws the bounding box around the detected object in green or red.
# Adds a label with confidence above the box.
    cv2.imshow("YOLO Detection", frame)
    #shows annotations of the frame in a window named YOLO Detection
    if cv2.waitKey(1) & 0xFF == ord('q'):
        #to quit, press q
        break

cap.release()
cv2.destroyAllWindows()
