import cv2
import dlib
import numpy as np
from imutils.video import VideoStream
from scipy.spatial import distance
import sqlite3
import csv
import datetime
import time
import os
import winsound  

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# Eye aspect ratio 
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
# Mouth aspect ratio 
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

EAR_THRESH = 0.25
EAR_CONSEC_FRAMES = 48
MAR_THRESH = 0.60
MAR_CONSEC_FRAMES = 15
VIDEO_DURATION = 60  


db_file = "drowsiness_alert_data.db"
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Create table if not exists
cursor.execute('''
CREATE TABLE IF NOT EXISTS states (
    timestamp TEXT,
    status TEXT,
    duration REAL
)
''')

video_dir = "videos"
os.makedirs(video_dir, exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# Initialize variables
frame_count = 0
drowsy_frames = 0
yawn_frames = 0
current_status = "ACTIVE"
start_time = time.time()
video_start_time = time.time()
video_writer = None
alert_beeping = False

def play_beep(sound_type):
    if sound_type == "drowsy":
        winsound.Beep(1000, 500)  
    elif sound_type == "alert":
        global alert_beeping
        alert_beeping = True
        while alert_beeping:
            winsound.Beep(1000, 500)  

def stop_alert_beep():
    global alert_beeping
    alert_beeping = False

def update_video_writer(status, frame_shape):
    global video_writer
    if video_writer:
        video_writer.release()
    video_path = os.path.join(video_dir, f"{status}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (frame_shape[1], frame_shape[0]))

vs = VideoStream(src=0).start()

while True:
    frame = vs.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
   
    faces = detector(gray)
    
    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        for face in faces:
            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            landmarks = predictor(gray, face)
            landmarks = np.array([(p.x, p.y) for p in landmarks.parts()])
            
            # Extract eye and mouth landmarks
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            mouth = landmarks[48:68]
            
            # landmarks 
            for (x, y) in landmarks:
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
            
            # Compute eye and mouth aspect ratios
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            mar = mouth_aspect_ratio(mouth)
            
            if ear < EAR_THRESH:
                drowsy_frames += 1
            else:
                drowsy_frames = 0
            
            if mar > MAR_THRESH:
                yawn_frames += 1
            else:
                yawn_frames = 0
            
            if drowsy_frames >= EAR_CONSEC_FRAMES:
                status = "DROWSY"
            elif yawn_frames >= MAR_CONSEC_FRAMES:
                status = "YAWNING"
            else:
                status = "ACTIVE"
            
            current_time = time.time()
            if status != current_status or (current_time - video_start_time) >= VIDEO_DURATION:
                if current_status != "ACTIVE":
                    duration = current_time - start_time
                    cursor.execute('''
                    INSERT INTO states (timestamp, status, duration)
                    VALUES (?, ?, ?)
                    ''', (datetime.datetime.now().isoformat(), current_status, duration))
                    conn.commit()
                
                update_video_writer(status, frame.shape)
                video_start_time = current_time
                if status == "DROWSY":
                    play_beep("drowsy")
                elif status == "ACTIVE":
                    stop_alert_beep()
                elif status == "YAWNING":
                    pass
                
                current_status = status
                start_time = current_time
            
            if video_writer:
                video_writer.write(frame)
            
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    cv2.imshow("Drowsiness Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  
        break

if current_status != "ACTIVE":
    duration = time.time() - start_time
    cursor.execute('''
    INSERT INTO states (timestamp, status, duration)
    VALUES (?, ?, ?)
    ''', (datetime.datetime.now().isoformat(), current_status, duration))
    conn.commit()
    
if video_writer:
    video_writer.release()


vs.stop()
cv2.destroyAllWindows()
conn.close()

# Export data to .CSVfile from sqLite db
with open('drowsiness_alert_data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Timestamp", "Status", "Duration"])
    
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM states')
    
    for row in cursor.fetchall():
        writer.writerow(row)
        
    conn.close()
