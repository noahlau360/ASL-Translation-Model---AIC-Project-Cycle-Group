# ASL-Translation-Model---AIC-Project-Cycle-Group
> **University of Washington AI Club (UW AIC)**  
> **2026 Project Cycles 3**

An end-to-end Computer Vision and Machine Learning system that translates American Sign Language (ASL) gestures into digital commands, bridging accessibility with hands-free smart home automation.

# Project Overview
This project builds an intuitive American Sign Language (ASL) translator powered by machine learning and expands its practical utility by integrating smart home device control.

Traditional smart home environments rely heavily on voice assistants (like Alexa or Google Home) or physical touchscreens, creating accessibility barriers for individuals who are deaf, hard of hearing, or non-verbal. By leveraging computer vision, this system interprets real-time ASL hand gestures to both translate human communication and directly control smart IoT devices (such as adjusting lighting, toggling switches, or triggering automation routines) without requiring a single spoken word or touch screen.

# Smart Home Integration & Use Cases
By mapping specific recognized ASL signs to IoT network triggers (e.g., via Home Assistant, MQTT, or local REST APIs), the platform turns natural hand signs into actionable device commands:
- Accessible Device Interaction: Enables deaf and non-verbal users to control their environment seamlessly using native sign language.
- Silent & Touchless Control: Adjust room lighting, media, or smart locks in noisy environments or situations where silent operation is preferred.
- Gesture-to-Action Mapping: Maps identified spatial feature coordinates to predefined action triggers:
  -  Example: Sign "OC" to Toggle smart lights or plugs ON/OFF.
  -  Custom Gestures: Trigger personalized home automation scenes (e.g., "Good Night" routine).
  
# Key Features
- Real-Time 3D Landmark Tracking: Utilizes Google MediaPipe HandLandmarker to extract 21 standard 3D hand joints ($x, y, z$ spatial coordinates), yielding a lightweight 63-feature representation per frame.
- Efficient Feature Preprocessing: Strips away background visual clutter and normalizes keypoint arrays into structured feature vectors for rapid model inference.
- ASL Machine Learning Pipeline: Trained classification model that evaluates feature vectors to output predicted ASL letters and signs along with confidence scores (predict_proba).
- IoT Action Dispatcher: Translates validated high-confidence sign predictions into smart home device actions.
- Model Serialization: Full persistence with joblib for re-loading trained estimators, feature schemas, and categorical label encoders during deployment.

# 🛠️ Tech Stack
- Language: Python 3
- Computer Vision: MediaPipe, OpenCV (cv2)
- Data & Machine Learning: NumPy, Pandas, Scikit-Learn, JoblibSmart Home / Networking: MQTT / Requests / Home Assistant API (Integration Layer)
- Visualization: Matplotlib

# Project Structure
├── data/                  # Sample input images and dataset files \
├── models/                # Serialized model (.pkl), feature schemas, and label encoders \
├── notebooks/             # UW AIC Colab notebooks for training & experimentation \
├── src/ \
│   ├── landmark_extractor.py  # MediaPipe vision parsing \
│   ├── predict.py             # Model inference & confidence scoring \
│   └── smart_home_bridge.py   # IoT command dispatcher / API handler \
├── README.md              # Project documentation \
└── requirements.txt       # Dependencies \

# Getting Started
1. InstallationClone the repository and install required packages: \
Bash \
git clone https://github.com/your-org/asl-smarthome-translator.git \
cd asl-smarthome-translator \
pip install -r requirements.txt \
3. Run Gesture Translation & Control \
Execute the pipeline to evaluate an image/frame and send the corresponding command to your smart home setup: \
Bash \
python src/predict.py --image_path path/to/image.jpg --trigger_smarthome True
