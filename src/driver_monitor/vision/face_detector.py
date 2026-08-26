from abc import ABC, abstractmethod
import cv2
import numpy as np
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
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self.mp = mp
        self.vision = vision
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
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb_frame)
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


class OpenCVHaarFaceDetector(BaseFaceDetector):
    def __init__(self):
        cascade_dir = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(cascade_dir + "haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier(cascade_dir + "haarcascade_eye.xml")

    def detect(self, packet: FramePacket) -> FaceObservation | None:
        gray = cv2.cvtColor(packet.frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        face_gray = gray[y:y + h, x:x + w]

        upper_face = face_gray[: int(h * 0.55), :]
        eyes = self.eye_cascade.detectMultiScale(upper_face, scaleFactor=1.1, minNeighbors=5, minSize=(18, 12))
        eyes = sorted(eyes, key=lambda box: box[0])[:2]

        if len(eyes) >= 2:
            left_eye_box, right_eye_box = eyes[0], eyes[1]
            left_eye = self._eye_landmarks(x, y, left_eye_box, closed=False)
            right_eye = self._eye_landmarks(x, y, right_eye_box, closed=False)
        else:
            left_eye = self._estimated_eye_landmarks(x, y, w, h, side="left", closed=True)
            right_eye = self._estimated_eye_landmarks(x, y, w, h, side="right", closed=True)

        mouth = self._mouth_landmarks(gray, x, y, w, h)
        nose = (x + w * 0.5, y + h * 0.50)

        return FaceObservation(
            bbox=(int(x), int(y), int(w), int(h)),
            landmarks={
                "left_eye": left_eye,
                "right_eye": right_eye,
                "mouth": mouth,
                "pose": [nose],
            },
            confidence=0.7,
            provider="opencv_haar",
        )

    def _eye_landmarks(self, face_x, face_y, eye_box, closed=False):
        ex, ey, ew, eh = eye_box
        return self._box_to_eye_landmarks(face_x + ex, face_y + ey, ew, eh, closed)

    def _estimated_eye_landmarks(self, x, y, w, h, side, closed=False):
        cx = x + (w * 0.35 if side == "left" else w * 0.65)
        cy = y + h * 0.36
        ew = w * 0.20
        eh = h * 0.035 if closed else h * 0.08
        return self._box_to_eye_landmarks(cx - ew / 2, cy - eh / 2, ew, eh, closed)

    def _box_to_eye_landmarks(self, x, y, w, h, closed=False):
        h = max(h * (0.25 if closed else 1.0), 2.0)
        cy = y + h / 2
        return [
            (x, cy),
            (x + w * 0.30, y),
            (x + w * 0.70, y),
            (x + w, cy),
            (x + w * 0.70, y + h),
            (x + w * 0.30, y + h),
        ]

    def _mouth_landmarks(self, gray, x, y, w, h):
        mx1 = int(x + w * 0.25)
        mx2 = int(x + w * 0.75)
        my1 = int(y + h * 0.58)
        my2 = int(y + h * 0.88)
        mouth_roi = gray[my1:my2, mx1:mx2]

        mouth_open_height = h * 0.06
        if mouth_roi.size > 0:
            blurred = cv2.GaussianBlur(mouth_roi, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 65, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                bx, by, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
                if bw > w * 0.12 and bh > h * 0.04:
                    mx1, mx2 = mx1 + bx, mx1 + bx + bw
                    center_y = my1 + by + bh / 2
                    mouth_open_height = bh
                    my1 = int(center_y - mouth_open_height / 2)
                    my2 = int(center_y + mouth_open_height / 2)

        left = (mx1, (my1 + my2) / 2)
        right = (mx2, (my1 + my2) / 2)
        top1 = ((mx1 + mx2) / 2, my1)
        bottom1 = ((mx1 + mx2) / 2, my2)
        top2 = (mx1 + (mx2 - mx1) * 0.35, my1)
        bottom2 = (mx1 + (mx2 - mx1) * 0.35, my2)
        return [left, right, top1, bottom1, top2, bottom2]
