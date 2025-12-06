import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# =============================
# Load FaceMesh sekali saja
# =============================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

# Landmark indeks (sesuaikan dengan file asli kamu)
LEFT_EYE_INDICES = [33, 246, 161, 160, 159, 158, 157]
RIGHT_EYE_INDICES = [263, 466, 388, 387, 386, 385, 384]


# =============================
# Fungsi: hitung titik pusat mata
# =============================
def get_eye_center(landmarks, image_shape, eye_indices):
    h, w = image_shape[:2]
    pts = []

    for idx in eye_indices:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        pts.append([x, y])

    pts = np.array(pts)
    center = np.mean(pts, axis=0)

    return np.array([int(center[0]), int(center[1])])


# =============================
# Crop ROI mata
# =============================
def crop_eye_roi(image, landmarks):
    h, w = image.shape[:2]
    pts = np.array([[int(l.x * w), int(l.y * h)] for l in landmarks])

    left_eye_pts = pts[LEFT_EYE_INDICES]
    right_eye_pts = pts[RIGHT_EYE_INDICES]

    eyes = np.vstack([left_eye_pts, right_eye_pts])

    x_min, y_min = eyes.min(axis=0)
    x_max, y_max = eyes.max(axis=0)

    margin_x = int((x_max - x_min) * 0.1)
    margin_y = int((y_max - y_min) * 1.75)

    x_min = max(0, x_min - margin_x)
    y_min = max(0, y_min - margin_y)
    x_max = min(w, x_max + margin_x)
    y_max = min(h, y_max + margin_y)

    eye_crop = image[y_min:y_max, x_min:x_max]

    if eye_crop.size == 0:
        return None

    return eye_crop


# =============================
# PREPROCESS UTAMA
# =============================
def preprocess_image(pil_image: Image.Image):
    # Convert PIL → OpenCV BGR
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark

    # ============ Hitung rotasi wajah ============
    left_eye = get_eye_center(landmarks, image.shape, LEFT_EYE_INDICES)
    right_eye = get_eye_center(landmarks, image.shape, RIGHT_EYE_INDICES)

    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    if abs(angle) > 2.0:
        h, w = image.shape[:2]
        eye_center = (
            float((left_eye[0] + right_eye[0]) / 2),
            float((left_eye[1] + right_eye[1]) / 2),
        )

        M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)

        aligned = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        aligned_landmarks = []
        for lm in landmarks:
            x = lm.x * w
            y = lm.y * h
            new_x = M[0, 0] * x + M[0, 1] * y + M[0, 2]
            new_y = M[1, 0] * x + M[1, 1] * y + M[1, 2]

            aligned_lm = type(
                "Landmark",
                (),
                {"x": new_x / w, "y": new_y / h},
            )()
            aligned_landmarks.append(aligned_lm)

        landmarks = aligned_landmarks
    else:
        aligned = image

    # ============ Crop area mata ============
    eye_roi = crop_eye_roi(aligned, landmarks)

    if eye_roi is None:
        return None

    # ============ Resize & Flatten (input KNN) ============
    eye_gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    eye_gray = cv2.resize(eye_gray, (48, 48))     # ukuran sesuai model kamu
    eye_gray = eye_gray / 255.0
    eye_flat = eye_gray.flatten().reshape(1, -1)

    return eye_flat
