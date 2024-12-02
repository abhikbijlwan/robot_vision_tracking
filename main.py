<<<<<<< HEAD

import torch
import torchvision
import numpy as np
import sys

print("Python search paths:", sys.path)

from sam2.build_sam import build_sam2_video_predictor

#path to the models
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"  
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"    

try:
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading SAM2 model: {e}")
    sys.exit(1)  # Exit if model loading fails


video_file = "Wallet.MOV" #video used for tracking 


prompts = {"points": [[100, 100], [200, 200]]}  #Two points in the frame

try:
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):

        state = predictor.init_state(video_file)

        frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, prompts)
        print(f"Processed frame {frame_idx}: Object IDs: {object_ids}")

        print("Propagating prompts through the video...")
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            print(f"Propagated frame {frame_idx}: Object IDs: {object_ids}")
         

except FileNotFoundError:
    print(f"Error: Video file '{video_file}' not found.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred during inference: {e}")
    sys.exit(1)

print("Processing completed successfully.")
=======
import cv2

video_path = "Wallet.MOV"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Couldn't open video.")
    exit()

tracker = cv2.TrackerCSRT.create() #Using CSRT Tracker from CV2

ret, frame = cap.read()

bbox = cv2.selectROI("Selection", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Selection")

#object tracking initialization
tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    success, bbox = tracker.update(frame)

    if success:
        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tracking failed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
cap.release()
>>>>>>> cb19ddf (Adding OpenCV Code)
