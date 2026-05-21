import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QTextEdit, QFrame, QLineEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap


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


# Fixed one-shot responses
COMMANDS: dict[str, str] = {}

# Commands that cycle through states on each call
TOGGLE_COMMANDS: dict[str, list[str]] = {
    "light": ["on", "off"],
}


class ASLTranslatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASL Translator")
        self.setMinimumSize(1050, 680)
        self.recording = False
        self.recorded_frames = []
        self._toggle_index: dict[str, int] = {}

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

    def _toggle_recording(self):
        if not self.recording:
            self.recording = True
            self.recorded_frames = []
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

        seconds = frame_count // 30
        placeholder = (
            f"[Connect your ASL recognition model here to populate this field.]\n"
            f"Session: {frame_count} frames captured (~{seconds}s of signing).\n"
        )
        self.output.append(placeholder)
        self.signs.setText("Model output will appear here")

    def _handle_input(self):
        text = self.cmd_input.text().strip().lower()
        if not text:
            return
        self.cmd_input.clear()

        if text in TOGGLE_COMMANDS:
            states = TOGGLE_COMMANDS[text]
            idx = self._toggle_index.get(text, 0)
            response = states[idx]
            self._toggle_index[text] = (idx + 1) % len(states)
        elif text in COMMANDS:
            response = COMMANDS[text]
        else:
            response = f'[unknown command: "{text}"]'

        print(f"INPUT:  {text}")
        print(f"OUTPUT: {response}")

        self.output.append(f"<b>&gt; {text}</b>")
        self.output.append(f"{response}\n")
        self.signs.setText(response)
        self.footer.setText(f'Command "{text}" → "{response}"')

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
