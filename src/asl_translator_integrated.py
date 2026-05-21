import sys
import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QTextEdit, QFrame, QLineEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
from collections import deque

# for switch control
import time
import asyncio
from dotenv import load_dotenv
from tplinkcloud import TPLinkDeviceManager

# intializing codes for smart switch
load_dotenv()
KASA_USER = os.getenv("KASA_USER")
KASA_PASS = os.getenv("KASA_PASS")
DEVICE_NAME = os.getenv("DEVICE_NAME")


class ASLRecognizer:
    """Handles ASL recognition using MediaPipe and trained model"""
    
    def __init__(self, model_path=os.path.join(os.path.dirname(__file__), 'classifier_improved_2.joblib')):
        self.model = None
        self.feature_columns = None
        self.label_encoder = None
        self.landmarker = None

        # Initialize MediaPipe
        self._init_mediapipe()
        
        # Load trained model
        self._load_model(model_path)
        
    def _init_mediapipe(self):
        """Initialize MediaPipe Hand Landmarker"""
        try:
            BaseOptions = python.BaseOptions
            HandLandmarker = vision.HandLandmarker
            HandLandmarkerOptions = vision.HandLandmarkerOptions
            VisionRunningMode = vision.RunningMode
            
            model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.IMAGE,
                num_hands=2
            )
            self.landmarker = HandLandmarker.create_from_options(options)
            print("✓ MediaPipe initialized successfully")
        except Exception as e:
            print(f"⚠ MediaPipe initialization failed: {e}")
            self.landmarker = None

    
    def _load_model(self, model_path):
        """Load the trained ASL classifier model"""
        try:
            loaded_data = joblib.load(model_path)
            if isinstance(loaded_data, dict):
                self.model = loaded_data['model']
                self.feature_columns = loaded_data['feature_columns']
                self.label_encoder = loaded_data.get('label_encoder')
            else:
                # Bare sklearn model — derive feature names and use classes_ directly
                self.model = loaded_data
                n = loaded_data.n_features_in_
                self.feature_columns = [
                    f"{axis}{i}" for i in range(n // 3) for axis in ('x', 'y', 'z')
                ]
                self.label_encoder = None
            print(f"✓ Model loaded from '{model_path}'")
        except FileNotFoundError:
            print(f"⚠ Model file '{model_path}' not found")
        except Exception as e:
            print(f"⚠ Error loading model: {e}")
    
    def extract_landmarks(self, frame):
        """Extract hand landmarks from a frame using MediaPipe"""
        if self.landmarker is None:
            return None

        try:
            # Convert to RGB and create MediaPipe Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect hands
            detection_result = self.landmarker.detect(mp_image)
            
            if not detection_result.hand_landmarks:
                return None
            
            # Get first hand's landmarks
            hand_landmarks = detection_result.hand_landmarks[0]
            
            # Extract x, y, z coordinates (63 features total) — raw, no normalization
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
            landmarks_flat = coords.flatten().tolist()
            return landmarks_flat if len(landmarks_flat) == 63 else None
            
        except Exception as e:
            print(f"Error extracting landmarks: {e}")
            return None
    
    def predict(self, landmarks):
        """Predict ASL sign from landmarks"""
        if self.model is None or landmarks is None:
            return None, 0.0, []
        
        try:
            import pandas as pd
            X = pd.DataFrame([landmarks], columns=self.feature_columns)
            probabilities = self.model.predict_proba(X)[0]
            certainty = np.max(probabilities) * 100

            classes = self.model.classes_
            predicted_label = classes[np.argmax(probabilities)]
            if self.label_encoder is not None:
                predicted_label = self.label_encoder.inverse_transform([np.argmax(probabilities)])[0]

            # Get top 3 predictions
            top_n = 3
            top_indices = np.argsort(probabilities)[::-1][:top_n]
            top_predictions = [
                (classes[idx] if self.label_encoder is None
                 else self.label_encoder.inverse_transform([idx])[0],
                 probabilities[idx] * 100)
                for idx in top_indices
            ]
            
            return predicted_label, certainty, top_predictions
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0, []


class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._running = False

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return
        self._running = True
        while self._running:
            ret, frame = cap.read()
            if ret:
                self.frame_ready.emit(frame)
        cap.release()

    def stop(self):
        self._running = False
        self.wait()

async def _kasa_control(action: str) -> bool:
    device_manager = TPLinkDeviceManager(KASA_USER, KASA_PASS)
    devices = await device_manager.get_devices()
    for d in devices:
        if d.get_alias() == DEVICE_NAME:
            if action == "on":
                await d.power_on()
            elif action == "off":
                await d.power_off()
            return True
    return False

class KasaThread(QThread):
    finished = pyqtSignal(str, bool)  # (action, success)

    def __init__(self, action: str):
        super().__init__()
        self.action = action

    def run(self):
        success = asyncio.run(_kasa_control(self.action))
        self.finished.emit(self.action, success)

# Command mappings
COMMANDS: dict[str, str] = {}

TOGGLE_COMMANDS: dict[str, list[str]] = {
    "light": ["on", "off"],
}

# Sign sequence triggers: map a tuple of signs (in order) to a terminal message
SIGN_SEQUENCES: dict[tuple, str] = {
    ("o", "n"): "Sequence detected: ON",
    ("o", "c"): "yay",
}


class ASLTranslatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASL Translator")
        self.setMinimumSize(1050, 680)
        self.recording = False
        self.recorded_frames = []
        self._toggle_index: dict[str, int] = {}

        self._last_command_time: dict[str, float] = {}
        self._kasa_thread: KasaThread | None = None
        
        # Initialize ASL recognizer
        self.recognizer = ASLRecognizer()
        
        # Buffer for averaging predictions
        self.prediction_buffer = deque(maxlen=5)

        # History of detected signs across recordings, for sequence matching
        max_seq_len = max((len(k) for k in SIGN_SEQUENCES), default=1)
        self.sign_history: deque[str] = deque(maxlen=max_seq_len)

        # Live real-time detection state
        self._live_frame_count = 0
        self._live_buffer: deque = deque(maxlen=18)   # rolling per-frame predictions
        self._last_stable_sign: str | None = None     # last sign that was "confirmed"
        self._stable_history: deque = deque(maxlen=2) # last 2 confirmed signs

        self._build_ui()
        self._apply_styles()

        self.camera = CameraThread()
        self.camera.frame_ready.connect(self._update_frame)
        self.camera.start()

    # ------------------------------------------------------------------ UI --

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(28, 24, 28, 20)
        layout.setSpacing(14)

        # Header row
        layout.addWidget(self._header())

        # Thin divider
        rule = QFrame()
        rule.setFrameShape(QFrame.Shape.HLine)
        rule.setObjectName("rule")
        layout.addWidget(rule)

        # Main content: camera left, output right
        content = QHBoxLayout()
        content.setSpacing(20)
        content.addWidget(self._camera_panel(), 3)
        content.addWidget(self._output_panel(), 2)
        layout.addLayout(content, stretch=1)

        # Command input bar
        layout.addWidget(self._input_bar())

        # Footer status
        self.footer = QLabel("Camera starting — please allow access if prompted.")
        self.footer.setObjectName("footer")
        layout.addWidget(self.footer)

    def _header(self):
        box = QWidget()
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)

        left = QVBoxLayout()
        title = QLabel("ASL Translator")
        title.setObjectName("title")
        sub = QLabel("Sign into the camera and press Record — translation appears on the right.")
        sub.setObjectName("sub")
        left.addWidget(title)
        left.addWidget(sub)

        row.addLayout(left)
        row.addStretch()
        return box

    def _camera_panel(self):
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # Panel title + live badge
        top = QHBoxLayout()
        lbl = QLabel("Camera Feed")
        lbl.setObjectName("panelTitle")
        top.addWidget(lbl)
        top.addStretch()
        self.badge = QLabel("● Live")
        self.badge.setObjectName("badgeLive")
        top.addWidget(self.badge)
        layout.addLayout(top)

        # Video display
        self.video = QLabel("Waiting for camera…")
        self.video.setObjectName("video")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(460, 340)
        layout.addWidget(self.video, stretch=1)

        # Live sign indicator
        live_row = QHBoxLayout()
        live_lbl = QLabel("Live Sign:")
        live_lbl.setObjectName("panelTitle")
        live_row.addWidget(live_lbl)
        self.live_sign_display = QLabel("—")
        self.live_sign_display.setObjectName("liveSignDisplay")
        live_row.addWidget(self.live_sign_display)
        live_row.addStretch()
        layout.addLayout(live_row)

        # Controls
        ctrl = QHBoxLayout()
        self.rec_btn = QPushButton("● Start Recording")
        self.rec_btn.setObjectName("recBtn")
        self.rec_btn.clicked.connect(self._toggle_recording)

        self.clear_btn = QPushButton("Clear Output")
        self.clear_btn.setObjectName("clearBtn")
        self.clear_btn.clicked.connect(self._clear)

        ctrl.addWidget(self.rec_btn)
        ctrl.addWidget(self.clear_btn)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        return panel

    def _output_panel(self):
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title = QLabel("Translation Output")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        self.output = QTextEdit()
        self.output.setObjectName("outputBox")
        self.output.setPlaceholderText(
            "Translation will appear here once you start recording and signing…"
        )
        self.output.setReadOnly(True)
        layout.addWidget(self.output, stretch=1)

        detected_title = QLabel("Last Detected Signs")
        detected_title.setObjectName("panelTitle")
        layout.addWidget(detected_title)

        self.signs = QLabel("—")
        self.signs.setObjectName("signsBox")
        self.signs.setWordWrap(True)
        self.signs.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.signs)

        return panel

    def _input_bar(self):
        bar = QFrame()
        bar.setObjectName("inputBar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(14, 10, 14, 10)
        row.setSpacing(10)

        lbl = QLabel("Command Input")
        lbl.setObjectName("inputBarLabel")
        row.addWidget(lbl)

        self.cmd_input = QLineEdit()
        self.cmd_input.setObjectName("cmdInput")
        self.cmd_input.setPlaceholderText('Type a command and press Enter (e.g. "light")')
        self.cmd_input.returnPressed.connect(self._handle_input)
        row.addWidget(self.cmd_input, stretch=1)

        send_btn = QPushButton("Send")
        send_btn.setObjectName("sendBtn")
        send_btn.clicked.connect(self._handle_input)
        row.addWidget(send_btn)

        return bar

    # --------------------------------------------------------------- slots --

    def _update_frame(self, frame: np.ndarray):
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(
            self.video.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video.setPixmap(pixmap)

        if self.recording:
            self.recorded_frames.append(frame.copy())

        # Live detection: sample every 3rd frame to avoid lag
        self._live_frame_count += 1
        if self._live_frame_count % 3 == 0:
            self._live_detect(frame)

    def _toggle_recording(self):
        if not self.recording:
            self.recording = True
            self.recorded_frames = []
            self.prediction_buffer.clear()
            self.rec_btn.setText("■ Stop Recording")
            self.rec_btn.setStyleSheet(
                "background-color:#EF4444;color:#fff;border-radius:8px;"
                "padding:10px 20px;font-size:13px;font-weight:600;"
            )
            self.badge.setText("● Recording")
            self.badge.setStyleSheet("color:#EF4444;font-size:12px;font-weight:600;")
            self.footer.setText("Recording in progress — sign into the camera.")
        else:
            self.recording = False
            count = len(self.recorded_frames)
            self.rec_btn.setText("● Start Recording")
            self.rec_btn.setStyleSheet("")  # revert to stylesheet
            self.badge.setText("● Live")
            self.badge.setStyleSheet("")
            self.footer.setText(f"Recording stopped — {count} frames captured.")
            self._run_translation(count)

    def _run_translation(self, frame_count: int):
        if frame_count < 5:
            self.output.append("[Clip too short — hold each sign for at least half a second.]\n")
            self.signs.setText("—")
            return

        # Process frames and detect signs
        detected_signs = []
        self.footer.setText("Processing frames...")

        sample_interval = max(1, frame_count // 20)  # sample up to ~20 frames

        for i in range(0, frame_count, sample_interval):
            frame = self.recorded_frames[i]
            landmarks = self.recognizer.extract_landmarks(frame)

            if landmarks is not None:
                predicted_label, certainty, _ = self.recognizer.predict(landmarks)
                print(f"  frame {i}: {predicted_label}  {certainty:.0f}%")
                if predicted_label and certainty > 50:
                    detected_signs.append((predicted_label, certainty))
        
        # Analyze detected signs
        if detected_signs:
            # Get most common sign
            from collections import Counter
            sign_counts = Counter([sign for sign, _ in detected_signs])
            most_common_sign, count = sign_counts.most_common(1)[0]
            avg_certainty = np.mean([cert for sign, cert in detected_signs if sign == most_common_sign])
            
            # Display results
            self.output.append(f"<b>Detected Sign: {most_common_sign}</b>")
            self.output.append(f"Confidence: {avg_certainty:.1f}%")
            self.output.append(f"Detected in {count}/{len(detected_signs)} analyzed frames\n")
            
            self.signs.setText(f"{most_common_sign} ({avg_certainty:.0f}%)")
            self.footer.setText(f"Translation complete: '{most_common_sign}' detected")

            self.sign_history.append(most_common_sign.lower())
            self._check_sign_sequences()
            
        else:
            self.output.append("[No hand signs detected with sufficient confidence.]\n")
            self.signs.setText("No clear sign detected")
            self.footer.setText("No signs detected — try again with clearer hand positioning")

    def _live_detect(self, frame: np.ndarray):
        landmarks = self.recognizer.extract_landmarks(frame)
        if landmarks is not None:
            label, certainty, _ = self.recognizer.predict(landmarks)
            if certainty > 35:
                self._live_buffer.append(label)
                self.live_sign_display.setText(f"{label.upper()}  {certainty:.0f}%")
            else:
                self._live_buffer.append(None)
                self.live_sign_display.setText("—")
        else:
            self._live_buffer.append(None)
            self.live_sign_display.setText("—")

        # Need a full buffer before judging stability
        if len(self._live_buffer) < self._live_buffer.maxlen:
            return

        from collections import Counter
        non_null = [s for s in self._live_buffer if s is not None]
        if not non_null:
            return
        top_sign, top_count = Counter(non_null).most_common(1)[0]

        # Sign is "stable" when it dominates ≥70% of the buffer
        if top_count / len(self._live_buffer) >= 0.55:
            if top_sign != self._last_stable_sign:
                self._last_stable_sign = top_sign
                self._stable_history.append(top_sign)
                self._live_buffer.clear()  # reset so next sign needs fresh frames
                self.sign_history.append(top_sign.lower())
                self._check_sign_sequences()
                self._check_double_sign()

    def _check_double_sign(self):
        history = list(self._stable_history)
        if len(history) == 2 and history[0] == history[1]:
            sign = history[0]
            print(f"DOUBLE SIGN: {sign} {sign} → ON")
            self.output.append(f"<b>[Double sign: {sign.upper()} {sign.upper()}]</b>")
            self.output.append("Triggered: ON\n")
            self.footer.setText(f"Double sign '{sign.upper()}' detected → ON")
            self._stable_history.clear()
            self._last_stable_sign = None

    def _check_sign_sequences(self):
        history = list(self.sign_history)
        for seq, message in SIGN_SEQUENCES.items():
            n = len(seq)
            if history[-n:] == list(seq):
                print(f"SIGN SEQUENCE: {' → '.join(seq)}")
                print(f"OUTPUT: {message}")
                self.output.append(f"<b>[Sign sequence: {' → '.join(seq)}]</b>")
                self.output.append(f"{message}\n")
                self.sign_history.clear()
                break

    def _handle_input(self):
        text = self.cmd_input.text().strip().lower()
        if not text:
            return
        self.cmd_input.clear()

        if text in TOGGLE_COMMANDS:
            cooldown = 2.0
            if time.time() - self._last_command_time.get(text, 0) < cooldown:
                return
            self._last_command_time[text] = time.time()

            states = TOGGLE_COMMANDS[text]
            idx = self._toggle_index.get(text, 0)
            response = states[idx]
            self._toggle_index[text] = (idx + 1) % len(states)

            if text == "light":
                self.footer.setText(f'Sending "{response}" to {DEVICE_NAME}…')
                self._kasa_thread = KasaThread(response)
                self._kasa_thread.finished.connect(self._on_kasa_done)
                self._kasa_thread.start()
        elif text in COMMANDS:
            response = COMMANDS[text]
        else:
            response = f'[unknown command: "{text}"]'

        print(f"INPUT:  {text}")
        print(f"OUTPUT: {response if text not in TOGGLE_COMMANDS or text != 'light' else '(kasa async)'}")

        if text != "light":
            self.output.append(f"<b>&gt; {text}</b>")
            self.output.append(f"{response}\n")
            self.signs.setText(response)
            self.footer.setText(f'Command "{text}" → "{response}"')
        else:
            self.output.append(f"<b>&gt; {text}</b>")
            self.output.append(f"Sending to device…\n")
            self.signs.setText("…")

    def _on_kasa_done(self, action: str, success: bool):
        if success:
            self.output.append(f"Light turned {action}.\n")
            self.signs.setText(action)
            self.footer.setText(f'Light turned {action}.')
        else:
            self.output.append(f'[Device "{DEVICE_NAME}" not found]\n')
            self.signs.setText("error")
            self.footer.setText(f'Could not reach "{DEVICE_NAME}".')

    def _clear(self):
        self.output.clear()
        self.signs.setText("—")
        self.footer.setText("Output cleared.")

    def closeEvent(self, event):
        self.camera.stop()
        event.accept()

    # ------------------------------------------------------------- styles --

    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #FFFFFF;
                font-family: 'Segoe UI', Arial, sans-serif;
            }

            QLabel#title {
                font-size: 26px;
                font-weight: 700;
                color: #111111;
                letter-spacing: -0.5px;
            }
            QLabel#sub {
                font-size: 13px;
                color: #888888;
            }

            QFrame#rule {
                border: none;
                background-color: #EBEBEB;
                max-height: 1px;
            }

            QFrame#panel {
                background-color: #FAFAFA;
                border: 1px solid #E5E5E5;
                border-radius: 14px;
            }
            QLabel#panelTitle {
                font-size: 12px;
                font-weight: 600;
                color: #555555;
                text-transform: uppercase;
                letter-spacing: 0.6px;
            }

            QLabel#video {
                background-color: #111111;
                border-radius: 10px;
                color: #666666;
                font-size: 13px;
            }

            QLabel#badgeLive {
                font-size: 12px;
                font-weight: 600;
                color: #22C55E;
            }

            QPushButton#recBtn {
                background-color: #111111;
                color: #FFFFFF;
                border: none;
                border-radius: 8px;
                padding: 10px 22px;
                font-size: 13px;
                font-weight: 600;
                min-width: 170px;
            }
            QPushButton#recBtn:hover {
                background-color: #2D2D2D;
            }

            QPushButton#clearBtn {
                background-color: #FFFFFF;
                color: #555555;
                border: 1px solid #D5D5D5;
                border-radius: 8px;
                padding: 10px 18px;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton#clearBtn:hover {
                background-color: #F4F4F4;
            }

            QTextEdit#outputBox {
                background-color: #FFFFFF;
                border: 1px solid #E5E5E5;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
                color: #222222;
                line-height: 1.6;
            }

            QLabel#liveSignDisplay {
                font-size: 18px;
                font-weight: 700;
                color: #111111;
                padding: 2px 8px;
            }

            QLabel#signsBox {
                background-color: #F2F2F2;
                border-radius: 8px;
                padding: 10px 14px;
                font-size: 16px;
                font-weight: 500;
                color: #111111;
                min-height: 40px;
            }

            QLabel#footer {
                font-size: 11px;
                color: #BBBBBB;
            }

            QFrame#inputBar {
                background-color: #F7F7F7;
                border: 1px solid #E5E5E5;
                border-radius: 10px;
            }
            QLabel#inputBarLabel {
                font-size: 12px;
                font-weight: 600;
                color: #555555;
                min-width: 110px;
            }
            QLineEdit#cmdInput {
                background-color: #FFFFFF;
                border: 1px solid #D5D5D5;
                border-radius: 7px;
                padding: 8px 12px;
                font-size: 13px;
                color: #111111;
            }
            QLineEdit#cmdInput:focus {
                border: 1px solid #111111;
            }
            QPushButton#sendBtn {
                background-color: #111111;
                color: #FFFFFF;
                border: none;
                border-radius: 7px;
                padding: 8px 20px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton#sendBtn:hover {
                background-color: #2D2D2D;
            }
        """)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ASLTranslatorApp()
    window.show()
    sys.exit(app.exec())
