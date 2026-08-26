import sys
import os
import tempfile
import platform
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import time
import shutil
import subprocess
import cv2
import numpy as np
from driver_monitor.core.contracts import FramePacket
from driver_monitor.vision.face_detector import MediaPipeFaceDetector, OpenCVHaarFaceDetector
from driver_monitor.vision.object_detector import DummyYOLOOnnxDetector
from driver_monitor.vision.pipeline import DriverSafetyPipeline
from driver_monitor.core.gaze_tracker import HeadPoseAndGazeAnalyzer

# Mapping danh sách Cảnh báo Tiếng Việt
ALERT_TRANSLATIONS = {
    "drowsy": "CANH BAO: NGUY CO BUON NGU!",
    "eyes_closed": "CANH BAO: NHAM MAT QUA LAU!",
    "yawning": "CANH BAO: NGAP!",
    "distracted": "CANH BAO: MAT TAP TRUNG!",
    "phone_use": "CANH BAO: DUNG DIEN THOAI!",
    "LOOK_DOWN_SUSTAINED": "CANH BAO: CUI NHIN DIEN THOAI (>2S)!",
    "LOOK_DOWN_BURST_3TIMES": "CANH BAO: THOI QUEN CUI NHIN (3+ LAN/10S)!"
}

VOICE_ALERT_MESSAGES = {
    "yawning": "cảnh báo có nguy cơ buồn ngủ",
    "eyes_closed": "cảnh báo, nguy hiểm nguy hiểm, ngủ gật ngủ gật, hãy tỉnh dậy ngay",
    "drowsy": "cảnh báo, nguy hiểm nguy hiểm, ngủ gật ngủ gật, hãy tỉnh dậy ngay",
    "distracted": "cảnh báo, bạn đang mất tập trung, hãy chú ý lái xe",
    "LOOK_DOWN_SUSTAINED": "cảnh báo, bạn đang mất tập trung, hãy chú ý lái xe",
    "LOOK_DOWN_BURST_3TIMES": "cảnh báo, bạn đang mất tập trung, hãy chú ý lái xe",
}


VOICE_ALERT_MESSAGES = {
    "yawning": "c\u1ea3nh b\u00e1o c\u00f3 nguy c\u01a1 bu\u1ed3n ng\u1ee7",
    "eyes_closed": "c\u1ea3nh b\u00e1o, nguy hi\u1ec3m nguy hi\u1ec3m, ng\u1ee7 g\u1eadt ng\u1ee7 g\u1eadt, h\u00e3y t\u1ec9nh d\u1eady ngay",
    "drowsy": "c\u1ea3nh b\u00e1o, nguy hi\u1ec3m nguy hi\u1ec3m, ng\u1ee7 g\u1eadt ng\u1ee7 g\u1eadt, h\u00e3y t\u1ec9nh d\u1eady ngay",
    "distracted": "c\u1ea3nh b\u00e1o, b\u1ea1n \u0111ang m\u1ea5t t\u1eadp trung, h\u00e3y ch\u00fa \u00fd l\u00e1i xe",
    "LOOK_DOWN_SUSTAINED": "c\u1ea3nh b\u00e1o, b\u1ea1n \u0111ang m\u1ea5t t\u1eadp trung, h\u00e3y ch\u00fa \u00fd l\u00e1i xe",
    "LOOK_DOWN_BURST_3TIMES": "c\u1ea3nh b\u00e1o, b\u1ea1n \u0111ang m\u1ea5t t\u1eadp trung, h\u00e3y ch\u00fa \u00fd l\u00e1i xe",
}


class UsbSpeakerVoiceAlert:
    def __init__(self, cooldown_sec=0.0):
        self.cooldown_sec = cooldown_sec
        self.last_spoken_at = {}
        self.process = None
        self.tts_cmd = shutil.which("espeak-ng") or shutil.which("espeak")
        self.powershell_cmd = shutil.which("powershell") or shutil.which("pwsh")
        self.edge_tts_cmd = shutil.which("edge-tts")
        self.audio_player_cmd = (
            shutil.which("mpg123")
            or shutil.which("ffplay")
            or shutil.which("mpv")
            or shutil.which("cvlc")
        )
        self.voice_cache_dir = Path(tempfile.gettempdir()) / "driver_monitor_voice_alerts"
        self.voice_cache_dir.mkdir(parents=True, exist_ok=True)
        self.windows_has_vietnamese_voice = self._has_windows_vietnamese_voice()

        if self.edge_tts_cmd and (self.audio_player_cmd or (sys.platform == "win32" and self.powershell_cmd)):
            print("[INFO] Se phat canh bao bang giong Viet edge-tts: vi-VN-HoaiMyNeural")
        elif sys.platform == "win32" and self.powershell_cmd and not self.tts_cmd and not self.windows_has_vietnamese_voice:
            print("[WARN] Windows chua co giong doc tieng Viet, se phat bang giong mac dinh. Muon doc chuan hon thi cai Vietnamese voice.")
        elif self.tts_cmd:
            print("[INFO] Se phat canh bao bang espeak-ng/espeak.")
        else:
            print("[WARN] Khong tim thay TTS/audio player. Tren Pi cai: sudo apt install mpg123 espeak-ng && python -m pip install edge-tts")

    def _has_windows_vietnamese_voice(self):
        if sys.platform != "win32" or not self.powershell_cmd:
            return False

        try:
            result = subprocess.run(
                [
                    self.powershell_cmd,
                    "-NoProfile",
                    "-Command",
                    "Add-Type -AssemblyName System.Speech; "
                    "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    "$voice = $speaker.GetInstalledVoices() | "
                    "Where-Object { $_.VoiceInfo.Culture.Name -eq 'vi-VN' } | "
                    "Select-Object -First 1; "
                    "if ($voice) { exit 0 } else { exit 1 }",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def speak(self, alert_key):
        message = VOICE_ALERT_MESSAGES.get(alert_key)
        if not message:
            return

        if self.process and self.process.poll() is None:
            return

        now = time.time()
        last_time = self.last_spoken_at.get(alert_key, 0.0)
        if self.cooldown_sec > 0 and now - last_time < self.cooldown_sec:
            return

        self.last_spoken_at[alert_key] = now

        try:
            if self.edge_tts_cmd and (self.audio_player_cmd or (sys.platform == "win32" and self.powershell_cmd)):
                audio_path = self._ensure_edge_tts_audio(alert_key, message)
                if audio_path:
                    self.process = self._play_audio_file(audio_path)
            elif self.tts_cmd:
                self.process = subprocess.Popen(
                    [self.tts_cmd, "-v", "vi", "-s", "145", "-a", "180", message],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif sys.platform == "win32" and self.powershell_cmd:
                env = os.environ.copy()
                env["DRIVER_MONITOR_TTS_MESSAGE"] = message
                self.process = subprocess.Popen(
                    [
                        self.powershell_cmd,
                        "-NoProfile",
                        "-Command",
                        "Add-Type -AssemblyName System.Speech; "
                        "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                        "$voice = $speaker.GetInstalledVoices() | "
                        "Where-Object { $_.VoiceInfo.Culture.Name -eq 'vi-VN' } | "
                        "Select-Object -First 1; "
                        "if ($voice) { $speaker.SelectVoice($voice.VoiceInfo.Name) }; "
                        "$speaker.Volume = 100; "
                        "$speaker.Rate = 0; "
                        "$speaker.Speak($env:DRIVER_MONITOR_TTS_MESSAGE)",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=env,
                )
            else:
                safe_message = message.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8")
                print(f"[VOICE ALERT] {safe_message}")
        except Exception as exc:
            print(f"[WARN] Khong phat duoc canh bao ra loa USB: {exc}")

    def _ensure_edge_tts_audio(self, alert_key, message):
        audio_path = self.voice_cache_dir / f"{alert_key}.mp3"
        if audio_path.exists() and audio_path.stat().st_size > 0:
            return audio_path

        print(f"[INFO] Dang tao file am thanh tieng Viet: {audio_path}")
        result = subprocess.run(
            [
                self.edge_tts_cmd,
                "--voice",
                "vi-VN-HoaiMyNeural",
                "--text",
                message,
                "--write-media",
                str(audio_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"[WARN] Tao giong Viet bang edge-tts that bai: {result.stderr.strip()}")
            return None

        return audio_path

    def _play_audio_file(self, audio_path):
        if self.audio_player_cmd:
            player_name = Path(self.audio_player_cmd).name.lower()
            if player_name == "ffplay":
                cmd = [self.audio_player_cmd, "-nodisp", "-autoexit", "-loglevel", "quiet", str(audio_path)]
            elif player_name in {"cvlc", "vlc"}:
                cmd = [self.audio_player_cmd, "--play-and-exit", "--quiet", str(audio_path)]
            else:
                cmd = [self.audio_player_cmd, str(audio_path)]

            return subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        if sys.platform != "win32" or not self.powershell_cmd:
            return None

        env = os.environ.copy()
        env["DRIVER_MONITOR_AUDIO_PATH"] = str(audio_path)
        return subprocess.Popen(
            [
                self.powershell_cmd,
                "-NoProfile",
                "-Command",
                "Add-Type -AssemblyName PresentationCore; "
                "$player = New-Object System.Windows.Media.MediaPlayer; "
                "$player.Open([Uri]::new($env:DRIVER_MONITOR_AUDIO_PATH)); "
                "Start-Sleep -Milliseconds 300; "
                "$player.Play(); "
                "Start-Sleep -Milliseconds 3500; "
                "$player.Close()",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )

# Quản lý thời gian duy trì các trạng thái (Timers)
class DrowsinessAndDistractionManager:
    def __init__(self, ear_thresh=0.20, mar_thresh=0.60, eye_time_thresh=1.2, yawn_time_thresh=1.0):
        self.EAR_THRESH = ear_thresh
        self.MAR_THRESH = mar_thresh
        self.EYE_TIME_THRESH = eye_time_thresh    # Nhắm mắt 1.2s -> Báo buồn ngủ
        self.YAWN_TIME_THRESH = yawn_time_thresh  # Há miệng ngáp 1.0s -> Báo ngáp ngay

        self.eye_closed_start = None
        self.yawn_start = None

    def update(self, ear_val, mar_val):
        current_time = time.time()
        active_alerts = {}

        # 1. Kiểm tra trạng thái Nhắm mắt / Buồn ngủ (EAR < 0.20)
        if 0.0 < ear_val < self.EAR_THRESH:
            if self.eye_closed_start is None:
                self.eye_closed_start = current_time
            eye_duration = current_time - self.eye_closed_start
            if eye_duration >= self.EYE_TIME_THRESH:
                active_alerts["drowsy"] = 1.0
                active_alerts["eyes_closed"] = 1.0
        else:
            self.eye_closed_start = None
            eye_duration = 0.0

        # 2. Kiểm tra trạng thái Ngáp (MAR > 0.60)
        if mar_val > self.MAR_THRESH:
            if self.yawn_start is None:
                self.yawn_start = current_time
            yawn_duration = current_time - self.yawn_start
            if yawn_duration >= self.YAWN_TIME_THRESH:
                active_alerts["yawning"] = 1.0
        else:
            self.yawn_start = None
            yawn_duration = 0.0

        return active_alerts, eye_duration, yawn_duration

def draw_dashboard(frame, processed, gaze_res, dms_alerts, eye_dur, yawn_dur):
    h, w, _ = frame.shape
    sidebar_w = 340
    canvas = np.zeros((h, w + sidebar_w, 3), dtype=np.uint8)
    canvas[:, :w] = frame

    # Sidebar Panel Background
    cv2.rectangle(canvas, (w, 0), (w + sidebar_w, h), (25, 25, 25), -1)
    cv2.line(canvas, (w, 0), (w, h), (80, 80, 80), 2)
    cv2.putText(canvas, "GIAM SAT TRONG THAI TAI XE", (w + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    metrics = getattr(processed, "metrics_data", {})

    # 1. BẢNG ĐO EAR (MẮT)
    ear_val = metrics.get("ear", 0.0)
    cv2.putText(canvas, f"Do mo mat (EAR): {ear_val:.2f} ({eye_dur:.1f}s)", (w + 15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    ear_color = (0, 0, 255) if 0.0 < ear_val < 0.20 else (0, 255, 0)
    cv2.rectangle(canvas, (w + 15, 72), (w + 15 + int(min(ear_val, 0.4)/0.4 * 300), 82), ear_color, -1)
    cv2.rectangle(canvas, (w + 15, 72), (w + 315, 82), (120, 120, 120), 1)

    # 2. BẢNG ĐO MAR (MIỆNG / NGÁP)
    mar_val = metrics.get("mar", 0.0)
    cv2.putText(canvas, f"Do mo mieng (MAR): {mar_val:.2f} ({yawn_dur:.1f}s)", (w + 15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    mar_color = (0, 0, 255) if mar_val > 0.60 else (0, 255, 0)
    cv2.rectangle(canvas, (w + 15, 112), (w + 15 + int(min(mar_val, 1.0) * 300), 122), mar_color, -1)
    cv2.rectangle(canvas, (w + 15, 112), (w + 315, 122), (120, 120, 120), 1)

    # 3. GÓC PITCH / YAW & VÙNG NHÌN (AOI)
    pitch, yaw = gaze_res["pitch"], gaze_res["yaw"]
    cv2.putText(canvas, f"Goc Pitch/Yaw: {pitch:.1f} / {yaw:.1f} deg", (w + 15, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    zone = gaze_res["zone"]
    zone_color = (0, 0, 255) if zone == "PHONE_DOWN" else ((0, 255, 255) if zone == "CALIBRATING..." else (0, 255, 0))
    cv2.putText(canvas, f"Vung nhin (AOI): {zone}", (w + 15, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.50, zone_color, 2)

    # 4. THỜI GIAN & TẦN SUẤT CÚI NHÌN
    dur_val = gaze_res["duration"] if zone == "PHONE_DOWN" else 0.0
    cv2.putText(canvas, f"Thoi gian cui nhin: {dur_val:.1f}s / 2.0s", (w + 15, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.rectangle(canvas, (w + 15, 208), (w + 15 + int(min(dur_val, 2.0)/2.0 * 300), 218), (0, 0, 255) if dur_val >= 2.0 else (0, 255, 0), -1)

    burst_count = gaze_res["burst_count"]
    b_color = (0, 0, 255) if burst_count >= 3 else (200, 200, 200)
    cv2.putText(canvas, f"Tan suat cui nhin (10s): {burst_count} lan / 3 lan", (w + 15, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.45, b_color, 1 if burst_count < 3 else 2)

    # 5. HỢP NHẤT ĐIỂM RỦI RO (NOISY-OR RISK SCORE)
    cv2.putText(canvas, "HOP NHAT RUI RO (Noisy-OR)", (w + 15, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    
    # Tập hợp tất cả tín hiệu vi phạm
    all_alerts = dict(dms_alerts)
    if gaze_res["alert_signal"]:
        all_alerts[gaze_res["alert_signal"]] = 1.0

    score = processed.risk_score
    if len(all_alerts) > 0:
        score = max(score, 0.90)  # Tự động đẩy điểm rủi ro lên 0.90 khi có bất kỳ vi phạm nào

    score_color = (0, 0, 255) if score >= 0.5 else (0, 255, 0)
    cv2.putText(canvas, f"Diem rui ro: {score:.2f}", (w + 15, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.75, score_color, 2)
    cv2.rectangle(canvas, (w + 15, 325), (w + 15 + int(score * 300), 340), score_color, -1)

    # Hướng dẫn
    cv2.putText(canvas, "Nhan 'c' de Reset goc nhin chuan", (w + 15, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

    # RENDER BÁO ĐỘNG ĐỎ KHI VI PHẠM
    y_alert = 405
    for sig in all_alerts:
        if sig in ALERT_TRANSLATIONS:
            cv2.putText(canvas, ALERT_TRANSLATIONS[sig], (w + 15, y_alert),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            y_alert += 25

    return canvas

def main():
    config = {
        "ear_threshold": 0.20,
        "mar_threshold": 0.60,
        "head_offset_threshold": 0.25,
        "drowsy_time_sec": 1.2,
        "distracted_time_sec": 2.0,
    }

    face_backend = os.environ.get("DRIVER_MONITOR_FACE_BACKEND", "auto").strip().lower()
    machine = platform.machine().lower()
    is_raspberry_pi = machine in {"armv7l", "aarch64", "arm64"}

    if face_backend == "mediapipe":
        print("[INFO] Ep dung MediaPipe theo DRIVER_MONITOR_FACE_BACKEND=mediapipe.")
        face_detector = MediaPipeFaceDetector("models/face_landmarker.task")
    elif face_backend == "opencv":
        print("[INFO] Ep dung OpenCV Haar theo DRIVER_MONITOR_FACE_BACKEND=opencv.")
        face_detector = OpenCVHaarFaceDetector()
    elif is_raspberry_pi:
        print("[INFO] Dang chay tren ARM/Raspberry Pi, dung OpenCV Haar de tranh loi MediaPipe illegal instruction.")
        print("[INFO] Neu Pi cua ban co MediaPipe tuong thich, chay: DRIVER_MONITOR_FACE_BACKEND=mediapipe python src/driver_monitor/main.py")
        face_detector = OpenCVHaarFaceDetector()
    else:
        face_detector = MediaPipeFaceDetector("models/face_landmarker.task")
    object_detector = DummyYOLOOnnxDetector()
    pipeline = DriverSafetyPipeline(face_detector, object_detector, config)
    gaze_analyzer = HeadPoseAndGazeAnalyzer()
    voice_alert = UsbSpeakerVoiceAlert(cooldown_sec=0.0)
    
    # Đặt ngưỡng thời gian ngáp = 1.0s, nhắm mắt = 1.2s
    dms_manager = DrowsinessAndDistractionManager(ear_thresh=0.20, mar_thresh=0.60, eye_time_thresh=1.2, yawn_time_thresh=1.0)

    cap = cv2.VideoCapture(0)
    frame_idx = 0

    print("[INFO] Dang chay Dashboard giam sat... Nhan 'v' de test loa, 'c' de Reset Calibrate, 'q' de thoat.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        img_h, img_w, _ = frame.shape
        packet = FramePacket(frame=frame, timestamp=time.time(), frame_index=frame_idx, fps=30.0)
        processed = pipeline.process(packet)

        # 1. Cập nhật chỉ số Gaze / Pitch / Yaw
        gaze_res = gaze_analyzer.process_from_face_obs(processed.face_obs, img_w, img_h)

        # 2. Cập nhật chỉ số Ngáp & Buồn ngủ (EAR/MAR Manager)
        ear_val = processed.metrics_data.get("ear", 0.0) if processed.metrics_data else 0.0
        mar_val = processed.metrics_data.get("mar", 0.0) if processed.metrics_data else 0.0
        dms_alerts, eye_dur, yawn_dur = dms_manager.update(ear_val, mar_val)

        active_voice_alerts = dict(dms_alerts)
        if processed.signals.get("distracted"):
            active_voice_alerts["distracted"] = 1.0
        if gaze_res["alert_signal"]:
            active_voice_alerts[gaze_res["alert_signal"]] = 1.0

        for alert_key in [
            "eyes_closed",
            "drowsy",
            "yawning",
            "distracted",
            "LOOK_DOWN_SUSTAINED",
            "LOOK_DOWN_BURST_3TIMES",
        ]:
            if active_voice_alerts.get(alert_key):
                voice_alert.speak(alert_key)
                break

        # 3. Vẽ Landmark lên Video Frame
        if processed.face_obs:
            x, y, w_box, h_box = processed.face_obs.bbox
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

            for eye_key in ["left_eye", "right_eye"]:
                for pt in processed.face_obs.landmarks.get(eye_key, []):
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (255, 0, 0), -1)

            for pt in processed.face_obs.landmarks.get("mouth", []):
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)

            nose_pt = gaze_res["nose_pt"]
            if nose_pt != (0, 0):
                cv2.circle(frame, nose_pt, 4, (0, 255, 255), -1)
                p_rad = np.radians(gaze_res["pitch"])
                y_rad = np.radians(gaze_res["yaw"])
                dir_x = int(nose_pt[0] + 60 * np.sin(y_rad))
                dir_y = int(nose_pt[1] + 60 * np.sin(p_rad))
                cv2.line(frame, nose_pt, (dir_x, dir_y), (0, 255, 255), 2)

        # 4. Vẽ Dashboard Sidebar
        dashboard = draw_dashboard(frame, processed, gaze_res, dms_alerts, eye_dur, yawn_dur)
        cv2.imshow("Driver Safety Monitor Dashboard", dashboard)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            gaze_analyzer.reset_calibration()
            print("[INFO] Da xoa goc cu, bat dau Calibrate lai goc nhin chuan...")
        elif key == ord('v'):
            voice_alert.speak("eyes_closed")
            print("[INFO] Dang phat thu canh bao am thanh...")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
