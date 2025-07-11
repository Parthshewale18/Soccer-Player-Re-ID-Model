# Soccer-Player-Re-ID-Model
This project focuses on re-identifying soccer players across two video feeds: a broadcast view and a tacticam view. The goal is to assign consistent player IDs between both views, even when players go out of view temporarily.

📁 Project Structure
player_reid_project/
├── videos/
│   ├── broadcast.mp4
│   └── tacticam.mp4
├── yolov11_model/
│   └── best.pt  # YOLOv11 trained weights
├── detections/
│   ├── broadcast/
│   └── tacticam/
├── tracking/
│   └── detect_and_track.py  # Shared script for both views
├── reid/
│   ├── reid_utils.py
│   └── match_players.py
├── player_mapping.json
├── README.md
└── requirements.txt

Setup Instructions
✅ 1. Python Version
Use Python 3.8 to 3.10. It's recommended to use a virtual environment:
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

✅ 2. Install Dependencies
Install all required libraries using:
pip install -r requirements.txt
If you don’t have the requirements.txt, you can manually install the key packages:
pip install torch torchvision opencv-python-headless scikit-learn ultralytics numpy tqdm

✅ 3. Place Model and Videos
Download and place your YOLOv11 model (trained .pt file) in yolov11_model/
Add your videos in the videos/ folder named:
tacticam.mp4
broadcast.mp4

🚀 How to Run the Code
🎥 1. Detection and Tracking
Run the detection + tracking code for both feeds:


python tracking/detect_and_track.py --video_path videos/tacticam.mp4 --output_dir detections/tacticam --model_path yolov11_model/best.pt
Then repeat for broadcast:


python tracking/detect_and_track.py --video_path videos/broadcast.mp4 --output_dir detections/broadcast --model_path yolov11_model/best.pt
Each run will:

Track players with DeepSORT

Save metadata in tracking_metadata.json

Save player crops in crops/

The script also supports resume mode so it won't start from frame 0 again.

🧠 2. Player Re-Identification
Once both videos are tracked:


python reid/match_players.py
This script:

Extracts embeddings using ResNet-50

Compares player crops across views

Matches and outputs a mapping file: player_mapping.json

📝 Example Output
json

 {
  "tacticam_to_broadcast": {
    "2": "1",
    "4": "23"
  }
}


🛠️ Troubleshooting
Slow or laggy tracking? → Ensure you're using a GPU. DeepSORT with ResNet-50 is heavy on CPU.

YOLO model not loading? → Double-check that the .pt file is correct and matches the Ultralytics version.

Missing crops? → Make sure tracking was completed successfully for both videos.

📚 Tools and Models Used
YOLOv11: For detecting players and balls.

DeepSORT: For assigning consistent IDs using motion and visual embeddings.

ResNet-50: For feature extraction in player Re-ID step.

✅ Final Note
This setup ensures that a player detected in one camera will be correctly matched with their identity in the other camera using deep features. It's particularly useful for sports analytics or coaching software.
