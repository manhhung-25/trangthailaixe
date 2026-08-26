import cv2
from ultralytics import YOLO

class PhoneDetector:
    def __init__(self, model_path="yolo11n.pt", conf_thresh=0.4):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        # COCO class 67: cell phone
        self.PHONE_CLASS_ID = 67

    def detect(self, frame):
        results = self.model(frame, verbose=False, conf=self.conf_thresh)[0]
        phone_detected = False
        boxes = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id == self.PHONE_CLASS_ID:
                phone_detected = True
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                boxes.append(xyxy)

        return {"phone_detected": phone_detected, "boxes": boxes}