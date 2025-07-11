from ultralytics import YOLO
import cv2
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import json

#For Tacticam
#load model
model = YOLO('yolov11_model/best.pt')
progress_file = 'progress_tacticam.txt'

# Load progress
start_frame = 0
try:
    with open(progress_file, 'r') as f:
        start_frame = int(f.read().strip())
except (ValueError, FileNotFoundError):
    start_frame = 0 

#load videos
video_path = 'videos/tacticam.mp4' 
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

#output folder
output_folder = 'detections/tacticam/'
os.makedirs(output_folder, exist_ok=True)

frame_num = start_frame
while cap.isOpened():
   success,frame = cap.read()
   if not success:
      break
   # Run YOLOv11 detection
   results = model(frame)
   for result in results:
       boxes = result.boxes
       for box in boxes:
           cls = int(box.cls[0])
           conf = float(box.conf[0])
           x1,y1,x2,y2 = map(int, box.xyxy[0])

           if cls == 0:
               crop = frame[y1:y2, x1:x2]
               crop_filename = os.path.join(output_folder,f"{frame_num}_{x1}_{y1}.jpg")
               cv2.imwrite(crop_filename, crop)
            
           #Draw bounding box
           cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
           
   with open(progress_file, 'w') as f:
        f.write(str(frame_num + 1))
   frame_num+=1
cap.release()
    
#-----------------------------------------------------------------------------

#for Broadcast
#load model
model = YOLO('yolov11_model/best.pt')
progress_file = 'progress_broadcast.txt'

# Load progress
start_frame = 0
try:
    with open(progress_file,'r') as f:
        start_frame = int(f.read().strip())
except (ValueError,FileNotFoundError):
    start_frame = 0

#load videos
video_path = 'videos/broadcast.mp4' 
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame)

#output folder
output_folder = 'detections/broadcast/'
os.makedirs(output_folder,exist_ok=True)

frame_num = start_frame
while cap.isOpened:
    success,frame = cap.read()
    if not success:
        break
    # Run YOLOv11 detection
    results = model(frame)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1,y1,x2,y2 = map(int,box.xyxy[0])

            if cls == 0:
               crop = frame[y1:y2, x1:x2]
               crop_filename = os.path.join(output_folder,f"{frame_num}_{x1}_{y1}.jpg")
               cv2.imwrite(crop_filename, crop)
            
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        
    with open(progress_file, 'w') as f:
        f.write(str(frame_num + 1))
    frame_num+=1
cap.release()

#--------------------------------------------------------------------------------

#Integrate YOLOv11 + DeepSORT
tracker = DeepSort(max_age=30)

# Input and output
video_path = 'videos/tacticam.mp4'
output_crop_dir = 'detections/tacticam/crops'
frame_log = 'detections/tacticam/last_frame.txt'
meta_path = 'detections/tacticam/tracking_metadata.json'

os.makedirs(output_crop_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if os.path.exists(frame_log):
    with open(frame_log,'r') as f:
        start_frame = int(f.read())
else:
    start_frame = 0

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
frame_num = start_frame

metadata = {}
if os.path.exists(meta_path):
    with open(meta_path,'r') as f:
        metadata = json.load(f)

while cap.isOpened():
    success,frame = cap.read()
    if not success:
        break
    detections = []
    results = model(frame)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == 0:  # Player class
                # [x, y, width, height], confidence, class
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'player'))

    # Update DeepSort tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = str(track.track_id)
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        # Save crop
        crop = frame[y1:y2, x1:x2]
        crop_filename = os.path.join(output_crop_dir, f"{frame_num}_{track_id}.jpg")
        cv2.imwrite(crop_filename, crop)

        # Update metadata
        metadata[f"{frame_num}_{track_id}"] = {
            "frame": frame_num,
            "track_id": track_id,
            "bbox": [x1, y1, x2, y2]
        }

        # Optional: Draw for debug
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Save latest frame number
    with open(frame_log, 'w') as f:
        f.write(str(frame_num))

    frame_num += 1

cap.release()

# Save metadata JSON
with open(meta_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print("✅ Tracking complete. Metadata saved.")

#--------------------------------------------------------------------------------


# -------- SETTINGS -------- #
video_path = 'videos/broadcast.mp4'
output_crop_dir = 'detections/broadcast/crops'
meta_path = 'detections/broadcast/tracking_metadata.json'
frame_log = 'detections/broadcast/last_frame.txt'

os.makedirs(output_crop_dir, exist_ok=True)


# -------- INIT TRACKER -------- #
tracker = DeepSort(max_age=30)

# -------- INIT VIDEO -------- #
cap = cv2.VideoCapture(video_path)

# Get last processed frame number
if os.path.exists(frame_log):
    with open(frame_log, 'r') as f:
        start_frame = int(f.read())
else:
    start_frame = 0

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
frame_num = start_frame

# -------- METADATA -------- #
metadata = {}
if os.path.exists(meta_path):
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

# -------- PROCESS VIDEO -------- #
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    detections = []
    results = model(frame)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == 0:  # Player class
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'player'))

    # Update DeepSort tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = str(track.track_id)
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        # Save crop
        crop = frame[y1:y2, x1:x2]
        crop_filename = os.path.join(output_crop_dir, f"{frame_num}_{track_id}.jpg")
        cv2.imwrite(crop_filename, crop)

        # Update metadata
        metadata[f"{frame_num}_{track_id}"] = {
            "frame": frame_num,
            "track_id": track_id,
            "bbox": [x1, y1, x2, y2]
        }

        # Optional: Draw for debug
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Save latest frame number
    with open(frame_log, 'w') as f:
        f.write(str(frame_num))

    frame_num += 1

cap.release()

# Save metadata JSON
with open(meta_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print("✅ Broadcast tracking complete. Metadata saved.")
