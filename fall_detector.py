import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import numpy as np

# Load the YOLOv8 model
model = YOLO("yolo11m.pt")
names=model.model.names

# Open the video file (use video file or webcam, here using webcam)
video = cv2.VideoCapture('c:/Users/ethomas308/Documents/CompVision/trip12fps.mp4')
count=0


while True:
    ret,frame = video.read()
    if not ret: #there are no more frames to process
        break 

    
    # if count % 2 != 0:
    #     continue #only process every 5th frame
    # count += 1 #current frame
    frame = cv2.resize(frame, (1020, 600))
    
    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True,classes=0)

    # Check if there are any boxes in the results
    result_frame = results[0].plot()

    if results[0].boxes is not None and results[0].boxes.id is not None:

        # Get the boxes (x, y, w, h), class IDs, track IDs, and confidences
        boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
        class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
        track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
        confidences = results[0].boxes.conf.cpu().tolist()  # Confidence score
       
        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = names[class_id]
            x1, y1, x2, y2 = box
            h = y2 - y1
            w = x2 - x1
            diff = h - w
            #print(diff)
            
            if diff <= 0: #if the box is wider than taller, bro fell
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                cvzone.putTextRect(frame,f'{track_id}',(x1,y2),1,1)
                cvzone.putTextRect(frame,f"{'Fall'}",(x1,y1),1,1)
                
            else:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cvzone.putTextRect(frame,f'{track_id}',(x1,y2),1,1)
                cvzone.putTextRect(frame,f"{'Normal'}",(x1,y1),1,1)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
       break

# Release the video capture object and close the display window
video.release()
cv2.destroyAllWindows()




# while True:
#    ret, frame = cap.read()
#    if not ret:
#        break  # Exit the loop if no frames are left to process
#    # Run pose estimation with tracking enabled
#    results = model.track(frame, task="pose")
#    # Visualize the tracked poses on the frame
#    result_frame = results[0].plot()  # Draw keypoints and bounding boxes
#    # Write the frame to the output video
#    out.write(result_frame)
# # Release resources
# cap.release()
# out.release()