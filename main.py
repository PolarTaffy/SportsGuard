import cv2
import cvzone
import numpy as np
from flask import Flask, Response, render_template_string
from ultralytics import YOLO
import time

app = Flask(__name__)

# Load the YOLO model
model = YOLO("yolo11s.pt")
names = model.model.names

# Open your video source (file or webcam)
source = 'C:\Users\ethomas308\Documents\GitHub\SportInjuryDetector/trip12fps.mp4'
video = cv2.VideoCapture(source)

#video = cv2.VideoCapture(0) # Webcam

#Event Log Initialization
global event_log
event_log = list()


def log_event(player_id, event_time, event_type):
    if (event_type == "Bump"):
        player1 = player_id[0]
        player2 = player_id[1]
        event_log.append(f"{event_type} detected between player {player1} and player {player2} at {event_time}")
    #fall_message = f"Fall detected for player {player_id} at {fall_time}"
    #event_log.append(fall_message)
                    
    event_log.append(f"{event_type} detected for player {player_id} at {event_time}")
    #update log event to include an image

def generate_frames():
    
    #event_log = list()

    while True:
        ret, frame = video.read()
        if not ret:
            break  # End of stream

        # Resize and process frame
        frame = cv2.resize(frame, (300, 200))
        results = model.track(frame, persist=True, classes=0)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            print("Results: ", results[0])

            boxes = results[0].boxes.xyxy.int().cpu().tolist()      # Bounding boxes
            class_ids = results[0].boxes.cls.int().cpu().tolist()     # Class IDs
            track_ids = results[0].boxes.id.int().cpu().tolist()      # Track IDs
            confidences = results[0].boxes.conf.cpu().tolist()        # Confidence scores

            # Process each detection
            for box, class_id, player_id, conf in zip(boxes, class_ids, track_ids, confidences):
                x1, y1, x2, y2 = box
                h = y2 - y1
                w = x2 - x1
                diff = h - w
                if diff <= 0:
                    fall_time = time.strftime("%H:%M:%S", time.localtime())
                    
                    log_event(player_id, fall_time, "Fall") 
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cvzone.putTextRect(frame, f'{player_id}', (x1, y2), scale=1, thickness=1)
                    cvzone.putTextRect(frame, "Fall", (x1, y1), scale=1, thickness=1)
                    
                    
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cvzone.putTextRect(frame, f'{player_id}', (x1, y2), scale=1, thickness=1)
                    cvzone.putTextRect(frame, "Normal", (x1, y1), scale=1, thickness=1)

            # Check for collisions
            for (box1, player1, conf1), (box2, player2, conf2) in zip(zip(boxes, track_ids, confidences), zip(boxes[1:], track_ids[1:], confidences[1:])):
                x1, y1, x2, y2 = box1
                x3, y3, x4, y4 = box2
                if x1 < x4 and x2 > x3 and y1 < y4 and y2 > y3:
                    bump_time = time.strftime("%H:%M:%S", time.localtime())
                    log_event((player1, player2), bump_time, "Bump")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                    cvzone.putTextRect(frame, f'{player1}', (x1, y2), scale=1, thickness=1)
                    cvzone.putTextRect(frame, f'{player2}', (x3, y4), scale=1, thickness=1)
                    cvzone.putTextRect(frame, "Bump", (x1, y1), scale=1, thickness=1)
                    cvzone.putTextRect(frame, "Bump", (x3, y3), scale=1, thickness=1)

                    
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

#share event log with the front end
@app.route('/event_log')
def get_event_log():
    return {'events': event_log}

# Simple page to display the video stream
@app.route('/')
def index():
    return render_template_string('''
    <html>
    <head>
        <title>Live Sports Injury Detector</title>
        <script>
            function fetchEventLog() {
                fetch('/event_log')
                    .then(response => response.json())
                    .then(data => {
                        const eventLogDiv = document.getElementById('event-log');
                        eventLogDiv.innerHTML = '';
                        data.events.forEach(event => {
                            const eventElement = document.createElement('div');
                            eventElement.textContent = event;
                            eventLogDiv.appendChild(eventElement);
                        });
                    });
            }

            setInterval(fetchEventLog, 3000); // Fetch event log every 3 seconds
        </script>
        <script type="module">
            // Import the functions you need from the SDKs you need
            import { initializeApp } from "https://www.gstatic.com/firebasejs/11.3.1/firebase-app.js";
            import { getAnalytics } from "https://www.gstatic.com/firebasejs/11.3.1/firebase-analytics.js";
            // TODO: Add SDKs for Firebase products that you want to use
            // https://firebase.google.com/docs/web/setup#available-libraries

            // Your web app's Firebase configuration
            // For Firebase JS SDK v7.20.0 and later, measurementId is optional
            const firebaseConfig = {
                apiKey: "AIzaSyCqaIVxfpWPFgBLAwI9p4b5c9s9pjmo5Do",
                authDomain: "sports-injury-detection.firebaseapp.com",
                projectId: "sports-injury-detection",
                storageBucket: "sports-injury-detection.firebasestorage.app",
                messagingSenderId: "559631184068",
                appId: "1:559631184068:web:95cb0d3d81b797f4125991",
                measurementId: "G-YDKS6XD0VW"
            };

            // Initialize Firebase
            const app = initializeApp(firebaseConfig);
            const analytics = getAnalytics(app);
        </script>
    </head>
    <body>
        <h1>Live Sports Injury Detector</h1>
        <p>Real-time detection of falls and collisions in sports</p>
        
        <div id=video-feed>
            <h2>Video Stream</h2>
            <img src="{{ url_for('video_feed') }}" style="width:1020px; height:auto;">
        </div>
        
        <div id="player-roster>
            <h2>Player Roster</h2>
        </div>
        
        <h2>Event Log</h2>
        <div id="event-log" style="height: 200px; overflow-y: scroll; border: 1px solid #ccc;"></div>

        <div id="debug-tools">
            <h3>Video Control</h3>
            <button onclick="TODO()">Restart Demo Video Feed</button>
            <button onclick="TODO()">Webcam Feed</button> 
            <input type="text" placeholder="Enter Video URL">
            <button>Submit</button>
                                  
            <h3>Device Sync</h3>
            <button onclick="TODO()">Manual Android Sync</button>      
        </div>
                        

    </body>
    </html>
    ''')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
