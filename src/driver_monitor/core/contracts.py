from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np

@dataclass
class FramePacket:
    frame: np.ndarray
    timestamp: float
    frame_index: int
    fps: float
    source_id: str = "default"

@dataclass
class FaceObservation:
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    landmarks: dict[str, list[tuple[float, float]]]  # 'left_eye', 'right_eye', 'mouth', 'pose'
    confidence: float
    provider: str

@dataclass
class ObjectObservation:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    provider: str

@dataclass
class DetectionEvent:
    timestamp: float
    signal: str
    state: str
    score: float
    severity: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class ProcessedFrame:
    frame_index: int
    timestamp: float
    risk_score: float
    signals: dict[str, float]
    events: list[DetectionEvent]
    latency_ms: float
    face_obs: Optional[FaceObservation] = None
    object_obs: list[ObjectObservation] = field(default_factory=list)