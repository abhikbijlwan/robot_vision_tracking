import torch
import torchvision
import numpy as np
import sys


print("Python search paths:", sys.path)

from sam2.build_sam import build_sam2_video_predictor


checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
# print(f"Using device: {device}")


try:
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)
    print("Model loaded successfully.")
except Exception as e:
    print(e)
    sys.exit(1)  

video_file = "Wallet.MOV"


prompts = {"points": [[100, 100], [200, 200]]} 

try:
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_file)

        frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, prompts)
        print(f"Processed frame {frame_idx}: Object IDs: {object_ids}")

        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            print(f"Propagated frame {frame_idx}: Object IDs: {object_ids}")

except FileNotFoundError:
    print(f"Video File not found.")
    sys.exit(1)
except Exception as e:
    print(e)
    sys.exit(1)
