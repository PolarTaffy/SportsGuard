import cv2
import cvzone
import numpy as np
from flask import Flask, Response, render_template_string
from ultralytics import YOLO

app = Flask(__name__)

# Load the YOLO model
model = YOLO("yolo11s.pt")
names = model.model.names

# Open your video source (file or webcam)
#video = cv2.VideoCapture('c:/Users/ethomas308/Documents/CompVision/trip12fps.mp4')
video = cv2.VideoCapture(0)

def generate_frames():
    while True:
        ret, frame = video.read()
        if not ret:
            break  # End of stream

        # Resize and process frame
        frame = cv2.resize(frame, (300, 200))
        results = model.track(frame, persist=True, classes=0)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()      # Bounding boxes
            class_ids = results[0].boxes.cls.int().cpu().tolist()     # Class IDs
            track_ids = results[0].boxes.id.int().cpu().tolist()      # Track IDs
            confidences = results[0].boxes.conf.cpu().tolist()        # Confidence scores

            # Process each detection
            for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
                x1, y1, x2, y2 = box
                h = y2 - y1
                w = x2 - x1
                diff = h - w
                if diff <= 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cvzone.putTextRect(frame, f'{track_id}', (x1, y2), scale=1, thickness=1)
                    cvzone.putTextRect(frame, "Fall", (x1, y1), scale=1, thickness=1)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cvzone.putTextRect(frame, f'{track_id}', (x1, y2), scale=1, thickness=1)
                    cvzone.putTextRect(frame, "Normal", (x1, y1), scale=1, thickness=1)
                    
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        # Yield frame in multipart MIME format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    if not video.isOpened():
        return Response("No video source!", status=200)

    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Simple page to display the video stream
@app.route('/')
def index():
    return render_template_string('''
    <html>
    <head>
        <title>Live Sports Injury Detector</title>
    </head>
    <body>
        <h1>Video Stream</h1>
        <img src="{{ url_for('video_feed') }}" style="width:1020px; height:auto;">
    </body>
    </html>
    ''')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
