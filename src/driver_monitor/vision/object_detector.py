from abc import ABC, abstractmethod
from driver_monitor.core.contracts import FramePacket, ObjectObservation

class BaseObjectDetector(ABC):
    @abstractmethod
    def detect(self, packet: FramePacket) -> list[ObjectObservation]:
        pass

class DummyYOLOOnnxDetector(BaseObjectDetector):
    """Mô phỏng ONNX YOLO Detector cho việc phát hiện phone_use."""
    def __init__(self, model_path: str = "models/driver-objects.onnx", threshold: float = 0.5):
        self.threshold = threshold
        # Thực tế sẽ load session: onnxruntime.InferenceSession(model_path)

    def detect(self, packet: FramePacket) -> list[ObjectObservation]:
        # Giả lập không bắt được vật thể nguy hiểm trong khung hình thô
        return []