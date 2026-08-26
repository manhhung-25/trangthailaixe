import time
from collections import deque
import cv2
import numpy as np

class HeadPoseAndGazeAnalyzer:
    def __init__(self):
        self.glance_history = deque()
        self.current_zone = "FORWARD"
        self.zone_start_time = time.time()
        self.last_look_down_state = False
        self.last_look_down_time = 0.0  # Chống dội tín hiệu (Debounce)

        # Hệ số làm mịn EMA
        self.smooth_pitch = 0.0
        self.smooth_yaw = 0.0
        self.alpha = 0.25  # Tăng độ mượt, giảm rung xóc signal

        # Calibration
        self.baseline_pitch = None
        self.baseline_yaw = None
        self.calib_samples_p = []
        self.calib_samples_y = []
        self.is_calibrated = False

    def reset_calibration(self):
        """Reset lại mốc góc nhìn thẳng chuẩn"""
        self.baseline_pitch = None
        self.baseline_yaw = None
        self.calib_samples_p.clear()
        self.calib_samples_y.clear()
        self.is_calibrated = False

    def process_from_face_obs(self, face_obs, img_w, img_h):
        current_time = time.time()

        if not face_obs or not hasattr(face_obs, "landmarks"):
            return {
                "pitch": 0.0, "yaw": 0.0, "zone": "UNKNOWN",
                "duration": 0.0, "burst_count": 0, "alert_signal": None,
                "nose_pt": (0, 0), "is_calibrated": self.is_calibrated
            }

        lms = face_obs.landmarks
        left_eye = lms.get("left_eye", [])
        right_eye = lms.get("right_eye", [])
        pose = lms.get("pose", [])

        if not left_eye or not right_eye:
            return {
                "pitch": 0.0, "yaw": 0.0, "zone": "UNKNOWN",
                "duration": 0.0, "burst_count": 0, "alert_signal": None,
                "nose_pt": (0, 0), "is_calibrated": self.is_calibrated
            }

        # 1. Tính tâm 2 mắt & chóp mũi
        l_center = np.mean(left_eye, axis=0)
        r_center = np.mean(right_eye, axis=0)
        eye_dist = np.linalg.norm(l_center - r_center)
        if eye_dist == 0:
            eye_dist = 1.0

        eye_midpoint = (l_center + r_center) / 2.0

        if len(pose) > 0:
            nose_pt = (int(pose[0][0]), int(pose[0][1]))
        else:
            nose_pt = (int(eye_midpoint[0]), int(eye_midpoint[1] + eye_dist * 0.35))

        # 2. Tính góc thô 2D
        dx = nose_pt[0] - eye_midpoint[0]
        dy = nose_pt[1] - eye_midpoint[1]

        raw_yaw = (dx / eye_dist) * 80.0
        raw_pitch = (dy / eye_dist) * 90.0

        # 3. AUTO-CALIBRATION (Tự lấy mốc nhìn thẳng trong 25 frames)
        if not self.is_calibrated:
            self.calib_samples_p.append(raw_pitch)
            self.calib_samples_y.append(raw_yaw)

            if len(self.calib_samples_p) >= 25:
                self.baseline_pitch = float(np.mean(self.calib_samples_p))
                self.baseline_yaw = float(np.mean(self.calib_samples_y))
                self.is_calibrated = True

            return {
                "pitch": 0.0, "yaw": 0.0, "zone": "CALIBRATING...",
                "duration": 0.0, "burst_count": 0, "alert_signal": None,
                "nose_pt": nose_pt, "is_calibrated": False
            }

        # 4. Tính góc chênh lệch thực tế so với mốc nhìn thẳng
        rel_pitch = raw_pitch - self.baseline_pitch
        rel_yaw = raw_yaw - self.baseline_yaw

        self.smooth_yaw = self.alpha * rel_yaw + (1 - self.alpha) * self.smooth_yaw
        self.smooth_pitch = self.alpha * rel_pitch + (1 - self.alpha) * self.smooth_pitch

        pitch = self.smooth_pitch
        yaw = self.smooth_yaw

        # 5. PHÂN VÙNG AOI VỚI NGƯỠNG MỚI (PITCH > 5.0 DEGREE)
        if pitch > 5.0:
            zone = "PHONE_DOWN"       # Cúi nhẹ/Hạ mắt xem điện thoại
        elif yaw < -18.0:
            zone = "MIRROR_LEFT"      # Nhìn gương trái
        elif yaw > 18.0:
            zone = "MIRROR_RIGHT"     # Nhìn gương phải
        elif pitch < -10.0:
            zone = "MIRROR_CENTER"    # Ngẩng mặt nhìn gương giữa
        else:
            zone = "FORWARD"          # Nhìn thẳng đường

        # 6. QUẢN LÝ THỜI GIAN & CHỐNG BÁO GIẢ TẦN SUẤT (DEBOUNCE)
        if zone != self.current_zone:
            self.current_zone = zone
            self.zone_start_time = current_time

        zone_duration = current_time - self.zone_start_time

        is_looking_down = (zone == "PHONE_DOWN")

        # Đếm tần suất cúi: Phải cách lần cúi trước tối thiểu 0.8 giây để tránh bị đếm dồn lặp
        if is_looking_down and not self.last_look_down_state:
            if (current_time - self.last_look_down_time) > 0.8:
                self.glance_history.append(current_time)
                self.last_look_down_time = current_time

        self.last_look_down_state = is_looking_down

        # Xóa bớt lịch sử cúi quá 10 giây
        while self.glance_history and (current_time - self.glance_history[0] > 10.0):
            self.glance_history.popleft()

        burst_count = len(self.glance_history)
        alert_signal = None

        # Quy tắc cảnh báo
        if is_looking_down and zone_duration >= 2.0:
            alert_signal = "LOOK_DOWN_SUSTAINED"
        elif burst_count >= 3:
            alert_signal = "LOOK_DOWN_BURST_3TIMES"

        return {
            "pitch": pitch,
            "yaw": yaw,
            "zone": zone,
            "duration": zone_duration,
            "burst_count": burst_count,
            "alert_signal": alert_signal,
            "nose_pt": nose_pt,
            "is_calibrated": True
        }