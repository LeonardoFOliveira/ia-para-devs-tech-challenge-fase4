# anomaly_detection_module.py

import cv2
import joblib
import os
import logging
import numpy as np
import mediapipe as mp

class AnomalyDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            project_root = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(project_root, 'models', 'anomaly_detector_model.pkl')
        try:
            data = joblib.load(model_path)
            self.model = data['model']
            self.scaler = data['scaler']
            self.pca = data['pca']
            logging.info("Modelo de detecção de anomalias carregado com sucesso.")
        except Exception as e:
            logging.error(f"Erro ao carregar o modelo de anomalias: {e}")
            raise e

        # Inicializar o Face Mesh do MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True)

    def extract_facial_landmarks(self, face_image):
        try:
            image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                vector = []
                for landmark in landmarks.landmark:
                    vector.extend([landmark.x, landmark.y, landmark.z])
                logging.debug("Marcos faciais extraídos com sucesso usando MediaPipe.")
                return vector
            else:
                logging.warning("Nenhum marco facial encontrado com MediaPipe.")
                return None
        except Exception as e:
            logging.error(f"Erro ao extrair marcos faciais com MediaPipe: {e}")
            return None

    def detect_anomaly(self, face_image):
        try:
            logging.debug("Iniciando detecção de anomalia.")

            # Verificar se a imagem de face é válida
            if face_image is None or face_image.size == 0:
                logging.warning("Imagem de face inválida ou vazia.")
                return False

            height, width = face_image.shape[:2]
            if height < 20 or width < 20:
                logging.warning(f"Imagem de face muito pequena: {width}x{height} pixels.")
                return False

            # Extrair marcos faciais usando MediaPipe
            vector = self.extract_facial_landmarks(face_image)
            if vector is None or len(vector) == 0:
                logging.warning("Não foi possível extrair marcos faciais ou vetor vazio.")
                return False

            # Converter o vetor em numpy array e ajustar a forma
            vector = np.array(vector).reshape(1, -1)

            # Aplicar o pré-processamento
            vector_scaled = self.scaler.transform(vector)
            vector_pca = self.pca.transform(vector_scaled)

            # Predizer anomalia
            prediction = self.model.predict(vector_pca)
            logging.debug(f"Predição de anomalia: {prediction}")

            if prediction[0] == -1:
                logging.info("Anomalia detectada (careta).")
                return True  # Anomalia detectada
            else:
                logging.debug("Nenhuma anomalia detectada.")
                return False

        except Exception as e:
            logging.error(f"Erro na detecção de anomalias: {e}")
            return False

    def close(self):
        self.face_mesh.close()
