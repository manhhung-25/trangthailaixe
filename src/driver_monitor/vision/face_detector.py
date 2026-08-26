from abc import ABC, abstractmethod
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from driver_monitor.core.contracts import FramePacket, FaceObservation

class BaseFaceDetector(ABC):
    @abstractmethod
    def detect(self, packet: FramePacket) -> FaceObservation | None:
        pass

class MediaPipeFaceDetector(BaseFaceDetector):
    # Các chỉ số index mốc landmark chuẩn cho mắt và miệng trên MediaPipe 468/478 points
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    MOUTH_INDICES = [61, 291, 0, 17, 81, 178] # left, right, top1, bottom1, top2, bottom2

    def __init__(self, model_path: str = "models/face_landmarker.task"):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, packet: FramePacket) -> FaceObservation | None:
        rgb_frame = cv2.cvtColor(packet.frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)

        if not detection_result.face_landmarks:
            return None

        face_landmarks = detection_result.face_landmarks[0]
        h, w, _ = packet.frame.shape

        # Chuyển đổi tọa độ chuẩn hóa (0..1) sang pixel (x, y)
        coords = [(lm.x * w, lm.y * h) for lm in face_landmarks]

        # Trích xuất nhóm mốc mắt và miệng
        left_eye = [coords[i] for i in self.LEFT_EYE_INDICES]
        right_eye = [coords[i] for i in self.RIGHT_EYE_INDICES]
        mouth = [coords[i] for i in self.MOUTH_INDICES]

        # Tính Bounding Box khuôn mặt
        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]
        bbox = (int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys)))

        landmarks = {
            "left_eye": left_eye,
            "right_eye": right_eye,
            "mouth": mouth,
            "pose": [coords[1]] # Mốc sống mũi (Nose tip)
        }

        return FaceObservation(
            bbox=bbox,
            landmarks=landmarks,
            confidence=0.9,
            provider="mediapipe"
        )