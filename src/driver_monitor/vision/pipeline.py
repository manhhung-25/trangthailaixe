import time
from driver_monitor.core.contracts import FramePacket, ProcessedFrame, DetectionEvent
from driver_monitor.core.metrics import calculate_ear, calculate_mar, calculate_head_offset
from driver_monitor.core.scoring import RiskScorer
from driver_monitor.vision.face_detector import BaseFaceDetector
from driver_monitor.vision.object_detector import BaseObjectDetector

class DriverSafetyPipeline:
    def __init__(self, face_detector: BaseFaceDetector, object_detector: BaseObjectDetector, config: dict):
        self.face_detector = face_detector
        self.object_detector = object_detector
        self.config = config
        self.scorer = RiskScorer()

        # Timers theo giay thuc (Timestamp tracking)
        self.eye_closed_start: float | None = None
        self.yawn_start: float | None = None
        self.distracted_start: float | None = None

    def process(self, packet: FramePacket) -> ProcessedFrame:
        start_time = time.perf_counter()
        now = packet.timestamp
        events = []
        
        # Datastructure luu tru gia tri do thuc te de hien thi dashboard
        metrics_data = {"ear": 0.0, "mar": 0.0, "head_offset": 0.0}
        signals = {
            "eyes_closed": 0.0, "drowsy": 0.0, "yawning": 0.0,
            "distracted": 0.0, "phone_use": 0.0, "driving_fatigue": 0.0,
            "cabin_occupant_risk": 0.0
        }

        face_obs = self.face_detector.detect(packet)
        object_obs = self.object_detector.detect(packet)

        if face_obs:
            # 1. EAR & Blink / Drowsy logic (Khong bao gia khi chop mat)
            left_ear = calculate_ear(face_obs.landmarks.get("left_eye", []))
            right_ear = calculate_ear(face_obs.landmarks.get("right_eye", []))
            avg_ear = (left_ear + right_ear) / 2.0 if (left_ear and right_ear) else left_ear or right_ear
            metrics_data["ear"] = round(avg_ear, 3)

            if 0.0 < avg_ear < self.config.get("ear_threshold", 0.20):
                if self.eye_closed_start is None:
                    self.eye_closed_start = now
                duration = now - self.eye_closed_start
                # Chi kich hoat khi nham mat lau hon chop mat thong thuong (>= 1.5s)
                if duration >= self.config.get("drowsy_time_sec", 1.5):
                    signals["drowsy"] = 1.0
                    events.append(DetectionEvent(now, "drowsy", "CRITICAL", 0.9, "HIGH", "Nham mat qua lau!"))
            else:
                self.eye_closed_start = None

            # 2. MAR & Yawning logic
            mar = calculate_mar(face_obs.landmarks.get("mouth", []))
            metrics_data["mar"] = round(mar, 3)

            if mar > self.config.get("mar_threshold", 0.60):
                if self.yawn_start is None:
                    self.yawn_start = now
                if (now - self.yawn_start) >= self.config.get("yawn_time_sec", 1.5):
                    signals["yawning"] = 1.0
            else:
                self.yawn_start = None

            # 3. Head Offset & Distracted (Chinh xac 5s moi bao dong)
            head_offset = calculate_head_offset(face_obs.landmarks)
            metrics_data["head_offset"] = round(head_offset, 3)

            if head_offset > self.config.get("head_offset_threshold", 0.22):
                if self.distracted_start is None:
                    self.distracted_start = now
                distracted_dur = now - self.distracted_start
                if distracted_dur >= self.config.get("distracted_time_sec", 5.0):
                    signals["distracted"] = 1.0
                    events.append(DetectionEvent(now, "distracted", "WARNING", 0.7, "MEDIUM", "Mat tap trung qua 5s!"))
            else:
                self.distracted_start = None

        # 4. Phone Use Signal
        for obj in object_obs:
            if obj.label in ["cell phone", "phone"] and obj.confidence > 0.5:
                signals["phone_use"] = 1.0

        # 5. Hop nhat rui ro qua Noisy-OR + Context Boost
        risk_score = self.scorer.calculate_risk_score(signals)
        latency = (time.perf_counter() - start_time) * 1000.0

        processed = ProcessedFrame(
            frame_index=packet.frame_index,
            timestamp=packet.timestamp,
            risk_score=risk_score,
            signals=signals,
            events=events,
            latency_ms=latency,
            face_obs=face_obs,
            object_obs=object_obs
        )
        # Gan metrics_data vao processed de ve dashboard
        processed.metrics_data = metrics_data
        processed.durations = {
            "eye_closed": round(now - self.eye_closed_start, 1) if self.eye_closed_start else 0.0,
            "distracted": round(now - self.distracted_start, 1) if self.distracted_start else 0.0,
        }
        return processed
