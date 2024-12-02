from deepface import DeepFace
import cv2
import logging

def analyze_emotions(frame, face_locations):
    emotions = []
    for top, right, bottom, left in face_locations:
        face_image = frame[top:bottom, left:right]
        try:
            analysis = DeepFace.analyze(
                face_image,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='skip'
            )

            # Verifica se 'analysis' é uma lista
            if isinstance(analysis, list):
                if len(analysis) > 0:
                    analysis = analysis[0]
                else:
                    raise ValueError("Análise retornou uma lista vazia.")

            emotion = analysis['dominant_emotion']
            emotions.append({'position': (left, top), 'emotion': emotion})
            logging.info(f"Resultado da análise de emoções: {analysis}")
        except Exception as e:
            logging.error(f"Erro na análise de emoções: {e}")
            emotions.append({'position': (left, top), 'emotion': 'unknown'})
    return emotions

def annotate_emotions(frame, emotions):
    for item in emotions:
        left, top = item['position']
        emotion = item['emotion']
        cv2.putText(frame, emotion, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return frame