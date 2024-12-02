import face_recognition
import joblib
import os
import logging
import numpy as np

class AnomalyDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            project_root = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(project_root, 'models', 'anomaly_detector_model.pkl')
        try:
            self.model = joblib.load(model_path)
            logging.info("Modelo de detecção de anomalias carregado com sucesso.")
        except Exception as e:
            logging.error(f"Erro ao carregar o modelo de anomalias: {e}")
            raise e

    def extract_facial_landmarks(self, face_image):
        try:
            logging.info("teste")
            face_landmarks_list = face_recognition.face_landmarks(face_image)
            logging.info(face_landmarks_list)
            if face_landmarks_list:
                logging.info("Marcos faciais extraídos com sucesso.")
                return face_landmarks_list[0]
            else:
                logging.warning("Nenhum marco facial encontrado.")
                return None
        except Exception as e:
            logging.error(f"Erro ao extrair marcos faciais: {e}")
            return None

    def landmarks_to_vector(self, landmarks):
        try:
            vector = []
            for key, points in landmarks.items():
                for point in points:
                    vector.extend(point)
            logging.info("Marcos faciais convertidos em vetor.")
            return vector
        except Exception as e:
            logging.error(f"Erro ao converter marcos em vetor: {e}")
            return None

    def detect_anomaly(self, face_image):
        try:
            logging.info("Iniciando detecção de anomalia.")

            # Verificar se a imagem de face é válida
            if face_image is None:
                logging.warning("Imagem de face é None.")
                return False

            if not isinstance(face_image, np.ndarray):
                logging.warning("Imagem de face não é um numpy.ndarray.")
                return False

            if face_image.size == 0:
                logging.warning("Imagem de face vazia.")
                return False

            height, width = face_image.shape[:2]
            if height < 20 or width < 20:
                logging.warning(f"Imagem de face muito pequena: {width}x{height} pixels.")
                return False

            # Extrair marcos faciais
            landmarks = self.extract_facial_landmarks(face_image)
            if landmarks is None:
                logging.warning("Não foi possível extrair marcos faciais.")
                return False

            # Converter marcos em vetor
            vector = self.landmarks_to_vector(landmarks)
            if vector is None or len(vector) == 0:
                logging.warning("Vetor de marcos faciais é vazio ou None.")
                return False

            # Garantir que o vetor esteja na forma adequada para o modelo
            vector = np.array(vector).reshape(1, -1)

            # Predizer anomalia
            prediction = self.model.predict(vector)
            logging.info(f"Predição de anomalia: {prediction}")

            if prediction[0] == -1:
                logging.info("Anomalia detectada (careta).")
                return True  # Anomalia detectada (careta)
            else:
                logging.debug("Nenhuma anomalia detectada.")
                return False

        except Exception as e:
            logging.error(f"Erro na detecção de anomalias: {e}")
            return False
