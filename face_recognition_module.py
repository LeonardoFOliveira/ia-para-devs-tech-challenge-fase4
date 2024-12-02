import cv2
import mediapipe as mp

# Inicialização do MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def detect_faces(frame):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        # Converter o frame para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Processar o frame para detectar faces
        results = face_detection.process(rgb_frame)
        face_locations = []

        if results.detections:
            for detection in results.detections:
                # Extrair a caixa delimitadora normalizada
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)

                # Adicionar as coordenadas (top, right, bottom, left)
                top = y
                right = x + w
                bottom = y + h
                left = x
                face_locations.append((top, right, bottom, left))

        return face_locations

def mark_faces(frame, face_locations):
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    return frame
