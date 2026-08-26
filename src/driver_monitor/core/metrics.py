import numpy as np

def calculate_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def calculate_ear(eye_landmarks: list[tuple[float, float]]) -> float:
    if len(eye_landmarks) < 6:
        return 0.0
    p1, p2, p3, p4, p5, p6 = eye_landmarks[:6]
    d_v1 = calculate_distance(p2, p6)
    d_v2 = calculate_distance(p3, p5)
    d_hor = calculate_distance(p1, p4)
    return (d_v1 + d_v2) / (2.0 * d_hor) if d_hor > 0 else 0.0

def calculate_mar(mouth_landmarks: list[tuple[float, float]]) -> float:
    if len(mouth_landmarks) < 6:
        return 0.0
    left, right = mouth_landmarks[0], mouth_landmarks[1]
    top1, bottom1 = mouth_landmarks[2], mouth_landmarks[3]
    top2, bottom2 = mouth_landmarks[4], mouth_landmarks[5]
    d_v1 = calculate_distance(top1, bottom1)
    d_v2 = calculate_distance(top2, bottom2)
    d_hor = calculate_distance(left, right)
    return (d_v1 + d_v2) / (2.0 * d_hor) if d_hor > 0 else 0.0

def calculate_head_offset(landmarks: dict[str, list[tuple[float, float]]]) -> float:
    """
    Tính độ lệch quay đầu (Head Offset).
    Nhìn thẳng => Offset xấp xỉ 0.0.
    Ngoảnh mặt sang trái/phải => Offset tăng lên > 0.35.
    """
    left_eye = landmarks.get("left_eye")
    right_eye = landmarks.get("right_eye")
    pose = landmarks.get("pose")

    if not left_eye or not right_eye or not pose:
        return 0.0

    # Lấy trung điểm 2 mắt
    mid_eyes_x = (left_eye[0][0] + right_eye[3][0]) / 2.0
    eye_dist = calculate_distance(left_eye[0], right_eye[3])

    if eye_dist == 0:
        return 0.0

    # Tọa độ X của đầu mũi
    nose_x = pose[0][0]

    # Độ lệch vị trí mũi so với trung điểm 2 mắt (chuẩn hóa theo khoảng cách mắt)
    offset = abs(nose_x - mid_eyes_x) / eye_dist
    return round(float(offset), 3)