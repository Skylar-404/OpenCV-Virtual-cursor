from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import mediapipe as mp
import pyautogui
import cv2
import numpy as np
import urllib.request
import signal
import time
import math
import os
import sys

# Force Qt to use X11/XWayland layer for clean window handling on Wayland desktop environments.
os.environ["QT_QPA_PLATFORM"] = "xcb"


# ---------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------
MODEL_FILE = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

CAM_WIDTH = 640
CAM_HEIGHT = 480

# Active box margin inside camera frame (creates deadzone to easily reach screen corners)
MARGIN_X = 90
MARGIN_Y = 70

# Pinch distance thresholds (in pixels on CAM_WIDTH x CAM_HEIGHT frame)
PINCH_THRESHOLD = 38     # Distance below which pinch / mouse-down is triggered
RELEASE_THRESHOLD = 52   # Distance above which pinch / mouse-up is released

# Cursor movement smoothing (lower value = faster response, higher = smoother)
SMOOTH_FACTOR = 3

# PyAutoGUI configuration fallback
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0


def ensure_model_exists():
    """Download the MediaPipe Hand Landmarker model asset if not already present."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_FILE)
    if not os.path.exists(model_path):
        print(f"[*] Downloading Hand Landmarker model to {model_path}...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("[+] Model download complete.")
    return model_path


# Hand Landmark Connections for drawing skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (0, 17)                                # Palm base
]


def draw_hand_landmarks_rgb(rgb_frame, landmarks_px):
    """Draw hand skeleton lines and joint landmarks directly onto RGB frame."""
    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx < len(landmarks_px) and end_idx < len(landmarks_px):
            pt1 = landmarks_px[start_idx]
            pt2 = landmarks_px[end_idx]
            cv2.line(rgb_frame, pt1, pt2, (200, 200, 200), 2)

    for idx, (x, y) in enumerate(landmarks_px):
        if idx in [4, 8, 12]:  # Thumb, Index, Middle tips
            cv2.circle(rgb_frame, (x, y), 8, (255, 255, 0), cv2.FILLED)
            cv2.circle(rgb_frame, (x, y), 10, (255, 255, 255), 1)
        else:
            cv2.circle(rgb_frame, (x, y), 4, (255, 120, 0), cv2.FILLED)


# ---------------------------------------------------------
# Low-Level Linux uinput Virtual Mouse Controller
# ---------------------------------------------------------
class MouseBackend:
    """
    Simulates hardware mouse movement, click-and-hold (dragging), and clicks.
    Uses Linux /dev/uinput for native hardware cursor rendering on Wayland.
    Falls back gracefully to PyAutoGUI if permissions are not yet configured.
    """

    def __init__(self):
        self.uinput_device = None
        self._is_left_down = False
        try:
            from evdev import UInput, ecodes as e
            capabilities = {
                e.EV_REL: [e.REL_X, e.REL_Y],
                e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT]
            }
            self.uinput_device = UInput(
                capabilities, name="GestureVirtualMouse")
            print(
                "[+] uinput virtual mouse initialized! (Native Wayland cursor active)")
        except Exception as ex:
            print(f"[*] uinput unavailable ({ex}). Using PyAutoGUI fallback.")
            print("[*] To activate native Ubuntu cursor, run this once in terminal:")
            print("    sudo chmod 666 /dev/uinput")

    def move(self, dx, dy, final_abs_x, final_abs_y):
        """Move mouse using uinput relative packets or PyAutoGUI absolute positioning."""
        if self.uinput_device is not None:
            from evdev import ecodes as e
            if dx != 0:
                self.uinput_device.write(e.EV_REL, e.REL_X, int(dx))
            if dy != 0:
                self.uinput_device.write(e.EV_REL, e.REL_Y, int(dy))
            self.uinput_device.syn()
        else:
            pyautogui.moveTo(final_abs_x, final_abs_y)

    def mouse_down(self, button='left'):
        """Hold down mouse button (for dragging/selecting)."""
        if button == 'left':
            self._is_left_down = True
        if self.uinput_device is not None:
            from evdev import ecodes as e
            btn_code = e.BTN_LEFT if button == 'left' else e.BTN_RIGHT
            self.uinput_device.write(e.EV_KEY, btn_code, 1)
            self.uinput_device.syn()
        else:
            pyautogui.mouseDown(button=button)

    def mouse_up(self, button='left'):
        """Release held mouse button."""
        if button == 'left':
            self._is_left_down = False
        if self.uinput_device is not None:
            from evdev import ecodes as e
            btn_code = e.BTN_LEFT if button == 'left' else e.BTN_RIGHT
            self.uinput_device.write(e.EV_KEY, btn_code, 0)
            self.uinput_device.syn()
        else:
            pyautogui.mouseUp(button=button)

    def click(self, button='right'):
        """Send a momentary click."""
        if self.uinput_device is not None:
            from evdev import ecodes as e
            btn_code = e.BTN_LEFT if button == 'left' else e.BTN_RIGHT
            self.uinput_device.write(e.EV_KEY, btn_code, 1)
            self.uinput_device.syn()
            time.sleep(0.015)
            self.uinput_device.write(e.EV_KEY, btn_code, 0)
            self.uinput_device.syn()
        else:
            pyautogui.click(button=button)

    def close(self):
        """Ensure mouse buttons are released and virtual device is closed."""
        if self._is_left_down:
            self.mouse_up('left')
        if self.uinput_device is not None:
            try:
                self.uinput_device.close()
            except Exception:
                pass


# ---------------------------------------------------------
# Camera Preview Window (Native PyQt6 HUD Window)
# ---------------------------------------------------------
class CameraPreviewWindow(QWidget):
    """
    Native PyQt6 window displaying the live OpenCV webcam HUD preview.
    """
    close_requested = pyqtSignal()

    def __init__(self, width=CAM_WIDTH, height=CAM_HEIGHT):
        super().__init__()
        self.setWindowTitle("Hand Gesture Cursor Control")
        self.setFixedSize(width, height)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.image_label)

    def update_frame(self, q_img):
        """Update window with new annotated camera frame."""
        self.image_label.setPixmap(QPixmap.fromImage(q_img))

    def keyPressEvent(self, event):
        """Allow exit via 'q' or 'ESC' keys."""
        if event.key() in (Qt.Key.Key_Q, Qt.Key.Key_Escape):
            print("[*] Exit key pressed in camera window.")
            self.close_requested.emit()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """Handle window close via 'X' button."""
        self.close_requested.emit()
        event.accept()


# ---------------------------------------------------------
# Gesture Tracking Thread (Optimized OpenCV + MediaPipe Video Mode)
# ---------------------------------------------------------
class GestureTrackingThread(QThread):
    """
    Background worker thread running optimized OpenCV video capture, MediaPipe Hand Landmarker
    in VIDEO tracking mode, and uinput/PyAutoGUI mouse simulation with Click & Hold (Dragging).
    """
    frame_ready = pyqtSignal(QImage)
    finished_tracking = pyqtSignal()

    def __init__(self, model_path, screen_w, screen_h):
        super().__init__()
        self.model_path = model_path
        self.screen_w = screen_w
        self.screen_h = screen_h
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        # Configure MediaPipe in VIDEO running mode for temporal tracking & higher FPS
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1
        )

        detector = None
        cap = None
        mouse = MouseBackend()

        try:
            detector = vision.HandLandmarker.create_from_options(options)

            print(
                f"[*] Target Screen Resolution: {self.screen_w}x{self.screen_h}")

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("[!] Error: Could not open webcam (/dev/video0).")
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Initialize cursor tracking from physical mouse position
            curr_pos = pyautogui.position()
            prev_screen_x, prev_screen_y = curr_pos.x, curr_pos.y

            is_left_down = False
            right_pinched = False
            hand_was_detected = False

            fps = 0.0
            prev_time = time.time()
            start_time = time.time()
            prev_timestamp_ms = -1

            while self._running:
                ret, frame = cap.read()
                if not ret:
                    print("[!] Failed to capture frame from webcam.")
                    break

                # Mirror frame horizontally
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                # Convert to RGB once for both MediaPipe and Qt GUI rendering
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Monotonically increasing timestamp for MediaPipe VIDEO mode
                curr_time = time.time()
                timestamp_ms = int((curr_time - start_time) * 1000)
                if timestamp_ms <= prev_timestamp_ms:
                    timestamp_ms = prev_timestamp_ms + 1
                prev_timestamp_ms = timestamp_ms

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = detector.detect_for_video(
                    mp_image, timestamp_ms)

                # Active box margin inside camera frame (draw directly on rgb_frame)
                cv2.rectangle(
                    rgb_frame,
                    (MARGIN_X, MARGIN_Y),
                    (w - MARGIN_X, h - MARGIN_Y),
                    (0, 100, 255),  # Orange in RGB
                    2
                )

                action_text = "Idle"
                action_color = (180, 180, 180)
                dist_thumb_index = 999.0
                dist_thumb_middle = 999.0

                if detection_result.hand_landmarks:
                    # Sync coordinates when hand re-enters to prevent sudden snap jumps
                    if not hand_was_detected:
                        curr_pos = pyautogui.position()
                        prev_screen_x, prev_screen_y = curr_pos.x, curr_pos.y
                        hand_was_detected = True

                    landmarks = detection_result.hand_landmarks[0]
                    landmarks_px = [
                        (int(lm.x * w), int(lm.y * h))
                        for lm in landmarks
                    ]

                    # Draw skeleton directly on RGB frame
                    draw_hand_landmarks_rgb(rgb_frame, landmarks_px)

                    thumb_x, thumb_y = landmarks_px[4]
                    index_x, index_y = landmarks_px[8]
                    middle_x, middle_y = landmarks_px[12]

                    is_index_up = landmarks[8].y < landmarks[6].y
                    is_middle_up = landmarks[12].y < landmarks[10].y

                    dist_thumb_index = math.hypot(
                        thumb_x - index_x, thumb_y - index_y)
                    dist_thumb_middle = math.hypot(
                        thumb_x - middle_x, thumb_y - middle_y)

                    # ---------------------------------------------------------
                    # Pinch Logic: Click & Hold (Drag) vs Right Click vs Move
                    # ---------------------------------------------------------
                    # Check if thumb & index are pinched (or already held down)
                    if dist_thumb_index < PINCH_THRESHOLD or (is_left_down and dist_thumb_index <= RELEASE_THRESHOLD):
                        # Action 1: Left Button DOWN (Click & Hold / Dragging)
                        if not is_left_down:
                            mouse.mouse_down('left')
                            is_left_down = True

                        # Visual indicator for pinch contact
                        pinch_center = ((thumb_x + index_x) //
                                        2, (thumb_y + index_y) // 2)
                        cv2.line(rgb_frame, (thumb_x, thumb_y),
                                 (index_x, index_y), (0, 255, 0), 3)
                        cv2.circle(rgb_frame, pinch_center,
                                   10, (0, 255, 0), cv2.FILLED)

                        # Tracking point follows pinch center during drag
                        track_x, track_y = pinch_center

                        # Calculate screen coordinates and move while holding button (Dragging)
                        norm_x = (track_x - MARGIN_X) / float(w - 2 * MARGIN_X)
                        norm_y = (track_y - MARGIN_Y) / float(h - 2 * MARGIN_Y)
                        norm_x = max(0.0, min(1.0, norm_x))
                        norm_y = max(0.0, min(1.0, norm_y))

                        target_screen_x = norm_x * self.screen_w
                        target_screen_y = norm_y * self.screen_h

                        curr_screen_x = prev_screen_x + \
                            (target_screen_x - prev_screen_x) / SMOOTH_FACTOR
                        curr_screen_y = prev_screen_y + \
                            (target_screen_y - prev_screen_y) / SMOOTH_FACTOR

                        final_x = int(
                            np.clip(curr_screen_x, 0, self.screen_w - 1))
                        final_y = int(
                            np.clip(curr_screen_y, 0, self.screen_h - 1))

                        dx = final_x - prev_screen_x
                        dy = final_y - prev_screen_y

                        mouse.move(dx, dy, final_x, final_y)
                        prev_screen_x, prev_screen_y = curr_screen_x, curr_screen_y

                        action_text = "DRAGGING (CLICK & HOLD)"
                        action_color = (0, 255, 0)  # Green

                    else:
                        # Pinch released: release left mouse button
                        if is_left_down:
                            mouse.mouse_up('left')
                            is_left_down = False

                        # Action 2: Thumb & Middle Pinch -> Right Click
                        if dist_thumb_middle < PINCH_THRESHOLD and dist_thumb_middle < dist_thumb_index:
                            cv2.line(rgb_frame, (thumb_x, thumb_y),
                                     (middle_x, middle_y), (255, 140, 0), 3)
                            cv2.circle(rgb_frame, ((
                                thumb_x + middle_x) // 2, (thumb_y + middle_y) // 2), 10, (255, 140, 0), cv2.FILLED)

                            if not right_pinched:
                                mouse.click('right')
                                right_pinched = True

                            action_text = "RIGHT CLICK"
                            action_color = (255, 140, 0)  # Amber
                        elif dist_thumb_middle > RELEASE_THRESHOLD:
                            right_pinched = False

                        # Action 3: Normal Cursor Movement (Index & Middle UP, Not Pinched)
                        if is_index_up and is_middle_up and not right_pinched:
                            mid_x = (index_x + middle_x) // 2
                            mid_y = (index_y + middle_y) // 2

                            cv2.line(rgb_frame, (index_x, index_y),
                                     (middle_x, middle_y), (0, 255, 255), 2)
                            cv2.circle(rgb_frame, (mid_x, mid_y),
                                       8, (0, 255, 255), 2)
                            cv2.circle(rgb_frame, (mid_x, mid_y),
                                       3, (0, 255, 255), cv2.FILLED)

                            norm_x = (mid_x - MARGIN_X) / \
                                float(w - 2 * MARGIN_X)
                            norm_y = (mid_y - MARGIN_Y) / \
                                float(h - 2 * MARGIN_Y)
                            norm_x = max(0.0, min(1.0, norm_x))
                            norm_y = max(0.0, min(1.0, norm_y))

                            target_screen_x = norm_x * self.screen_w
                            target_screen_y = norm_y * self.screen_h

                            curr_screen_x = prev_screen_x + \
                                (target_screen_x - prev_screen_x) / SMOOTH_FACTOR
                            curr_screen_y = prev_screen_y + \
                                (target_screen_y - prev_screen_y) / SMOOTH_FACTOR

                            final_x = int(
                                np.clip(curr_screen_x, 0, self.screen_w - 1))
                            final_y = int(
                                np.clip(curr_screen_y, 0, self.screen_h - 1))

                            dx = final_x - prev_screen_x
                            dy = final_y - prev_screen_y

                            mouse.move(dx, dy, final_x, final_y)
                            prev_screen_x, prev_screen_y = curr_screen_x, curr_screen_y

                            action_text = "MOVING CURSOR"
                            action_color = (0, 255, 255)  # Cyan

                    # HUD Distance readouts
                    cv2.putText(
                        rgb_frame,
                        f"T-Index Dist: {int(dist_thumb_index)}px",
                        (15, h - 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0) if dist_thumb_index < PINCH_THRESHOLD else (
                            220, 220, 220),
                        1
                    )
                    cv2.putText(
                        rgb_frame,
                        f"T-Middle Dist: {int(dist_thumb_middle)}px",
                        (15, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 140, 0) if dist_thumb_middle < PINCH_THRESHOLD else (
                            220, 220, 220),
                        1
                    )

                else:
                    # Hand left camera view: release held drag if active
                    if is_left_down:
                        mouse.mouse_up('left')
                        is_left_down = False
                    hand_was_detected = False
                    right_pinched = False

                # FPS calculation
                fps = 1.0 / (curr_time - prev_time) if (curr_time -
                                                        prev_time) > 0 else 30.0
                prev_time = curr_time

                # HUD overlays directly on RGB frame
                cv2.putText(rgb_frame, f"Action: {action_text}", (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, action_color, 2)
                cv2.putText(rgb_frame, f"FPS: {int(fps)}", (w - 110, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(rgb_frame, "Move: Index+Middle UP | Drag/Click: Thumb+Index | R-Click: Thumb+Middle | Q: Quit",
                            (10, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

                # Emit QImage directly from rgb_frame buffer (zero redundant conversions)
                q_img = QImage(
                    rgb_frame.data,
                    w,
                    h,
                    3 * w,
                    QImage.Format.Format_RGB888
                ).copy()
                self.frame_ready.emit(q_img)

        finally:
            if cap is not None and cap.isOpened():
                cap.release()
            if detector is not None:
                detector.close()
            mouse.close()
            self.finished_tracking.emit()


def main():
    # Enable clean Ctrl+C termination from terminal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    print("=" * 60)
    print("      Optimized Hand Gesture Mouse Controller")
    print("      with Click & Hold Dragging (uinput)")
    print("=" * 60)
    print("Actions:")
    print(" 1. Move Cursor:        Both Index & Middle fingers UP")
    print(" 2. Click & Hold / Drag: Thumb & Index pinch (move while pinched)")
    print(" 3. Quick Left Click:   Quick Thumb & Index pinch & release")
    print(" 4. Right Click:        Thumb & Middle pinch")
    print("Controls:")
    print(" - Press 'q' or 'ESC' or close the preview window to exit")
    print(" - Press Ctrl+C in terminal to exit")
    print("=" * 60)

    model_path = ensure_model_exists()

    app = QApplication(sys.argv)

    # Automatically target primary monitor (1920x1080)
    primary_screen = app.primaryScreen()
    if primary_screen:
        screen_geo = primary_screen.geometry()
        screen_w, screen_h = screen_geo.width(), screen_geo.height()
    else:
        screen_w, screen_h = 1920, 1080

    # Periodic timer to keep Python interpreter processing POSIX signals
    sig_timer = QTimer()
    sig_timer.start(500)
    sig_timer.timeout.connect(lambda: None)

    # Native camera HUD preview window
    preview_window = CameraPreviewWindow()
    preview_window.show()

    # Gesture tracking background thread
    tracking_thread = GestureTrackingThread(model_path, screen_w, screen_h)
    tracking_thread.frame_ready.connect(preview_window.update_frame)
    tracking_thread.finished_tracking.connect(app.quit)
    preview_window.close_requested.connect(tracking_thread.stop)

    tracking_thread.start()

    # Run Qt application event loop
    app.exec()

    # Ensure thread cleans up on exit
    tracking_thread.stop()
    tracking_thread.wait()
    print("[*] Hand gesture mouse controller closed cleanly.")


if __name__ == "__main__":
    main()
