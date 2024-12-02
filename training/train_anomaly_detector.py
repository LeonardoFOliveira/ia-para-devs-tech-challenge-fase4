# train_anomaly_detector.py

import cv2
import face_recognition
from sklearn.ensemble import IsolationForest
import joblib
import os

def extract_facial_landmarks(face_image):
    face_landmarks_list = face_recognition.face_landmarks(face_image)
    if face_landmarks_list:
        return face_landmarks_list[0]
    else:
        return None

def landmarks_to_vector(landmarks):
    vector = []
    for key, points in landmarks.items():
        for point in points:
            vector.extend(point)
    return vector

def collect_normal_data(video_path, num_samples=500):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    normal_vectors = []

    while len(normal_vectors) < num_samples:
        ret, frame = cap.read()
        if not ret:
            break  # Fim do vídeo

        frame_count += 1
        # Processar apenas a cada N frames para acelerar
        if frame_count % 10 != 0:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model='hog')

        for top, right, bottom, left in face_locations:
            face_image = rgb_frame[top:bottom, left:right]
            landmarks = extract_facial_landmarks(face_image)
            if landmarks:
                vector = landmarks_to_vector(landmarks)
                normal_vectors.append(vector)
                print(f"Coletados {len(normal_vectors)} de {num_samples} exemplos.")

                # Parar se atingirmos o número desejado de amostras
                if len(normal_vectors) >= num_samples:
                    break

    cap.release()
    return normal_vectors

def main():
    # Caminho para o vídeo com expressões faciais normais
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    models_dir = os.path.join(project_root, 'models')
    
    # Caminho para o vídeo de treinamento
    normal_video_path = os.path.join(data_dir, 'video_normal.mp4')

    # Verifica se o arquivo existe
    if not os.path.exists(normal_video_path):
        print(f"O arquivo {normal_video_path} não foi encontrado.")
        print("Por favor, coloque um vídeo com expressões normais denominado 'video_normal.mp4' na pasta do projeto.")
        return

    print("Coletando dados de expressões faciais normais...")
    normal_vectors = collect_normal_data(normal_video_path, num_samples=500)

    # Treinamento do modelo IsolationForest
    print("Treinando o modelo de detecção de anomalias...")
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(normal_vectors)

    # Salvar o modelo treinado
    # Caminho para salvar o modelo treinado
    model_filename = os.path.join(models_dir, 'anomaly_detector_model.pkl')
    joblib.dump(model, model_filename)
    print(f"Modelo treinado salvo em '{model_filename}'.")

if __name__ == '__main__':
    main()
