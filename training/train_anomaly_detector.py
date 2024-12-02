import cv2
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import mediapipe as mp
import numpy as np
import logging

def extract_facial_landmarks(face_image, face_mesh):
    try:
        image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            vector = []
            for landmark in landmarks.landmark:
                vector.extend([landmark.x, landmark.y, landmark.z])
            return vector
        else:
            return None
    except Exception as e:
        logging.error(f"Erro ao extrair marcos faciais com MediaPipe: {e}")
        return None

def collect_normal_data(video_path, num_samples=500):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    normal_vectors = []

    # Inicializar o Face Mesh do MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    mp_face_detection = mp.solutions.face_detection

    while len(normal_vectors) < num_samples:
        ret, frame = cap.read()
        if not ret:
            break  # Fim do vídeo

        frame_count += 1
        # Processar apenas a cada N frames para acelerar
        if frame_count % 6 != 0:
            continue

        # Converter o frame para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Detectar faces usando MediaPipe Face Detection
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    # Extrair a caixa delimitadora normalizada
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x = int(bboxC.xmin * iw)
                    y = int(bboxC.ymin * ih)
                    w = int(bboxC.width * iw)
                    h = int(bboxC.height * ih)

                    # Extrair a imagem da face
                    face_image = frame[y:y+h, x:x+w]

                    # Verificar se a imagem de face é válida
                    if face_image is None or face_image.size == 0:
                        continue

                    # Extrair marcos faciais
                    vector = extract_facial_landmarks(face_image, face_mesh)
                    if vector is not None:
                        normal_vectors.append(vector)
                        print(f"Coletados {len(normal_vectors)} de {num_samples} exemplos.")

                        # Parar se atingirmos o número desejado de amostras
                        if len(normal_vectors) >= num_samples:
                            break

    cap.release()
    face_mesh.close()
    return normal_vectors

def main():
    # Caminho para o vídeo com expressões faciais normais
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    normal_video_path = os.path.join(data_dir, 'video_normal.mp4')

    # Verifica se o arquivo existe
    if not os.path.exists(normal_video_path):
        print(f"O arquivo {normal_video_path} não foi encontrado.")
        print("Por favor, coloque um vídeo com expressões normais denominado 'video_normal.mp4' na pasta 'data'.")
        return

    print("Coletando dados de expressões faciais normais...")
    normal_vectors = collect_normal_data(normal_video_path, num_samples=500)

    # Converter a lista de vetores em um numpy array
    normal_vectors = np.array(normal_vectors)

    # Pré-processamento dos dados
    scaler = StandardScaler()
    normal_vectors_scaled = scaler.fit_transform(normal_vectors)

    # Reduzir a dimensionalidade com PCA
    pca = PCA(n_components=50)  # Ajuste o número de componentes conforme necessário
    normal_vectors_pca = pca.fit_transform(normal_vectors_scaled)

    # Treinamento do modelo IsolationForest
    print("Treinando o modelo de detecção de anomalias...")
    model = IsolationForest(contamination=0.01, n_estimators=500, random_state=42)
    model.fit(normal_vectors_pca)

    # Salvar o modelo treinado e os objetos de pré-processamento
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_filename = os.path.join(models_dir, 'anomaly_detector_model.pkl')
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'pca': pca
    }, model_filename)
    print(f"Modelo treinado salvo em '{model_filename}'.")

if __name__ == '__main__':
    main()
