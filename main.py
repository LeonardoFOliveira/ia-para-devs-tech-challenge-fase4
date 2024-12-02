from video_processor import VideoProcessor
from face_recognition_module import detect_faces, mark_faces
from emotion_analysis_module import analyze_emotions, annotate_emotions
from activity_detection_module import ActivityDetector
from anomaly_detection_module import AnomalyDetector
from summary_generator import SummaryGenerator
import cv2
import os
import logging

#Configuração básica do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Iniciando o processamento do vídeo.")
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, 'data')
    video_path = os.path.join(data_dir, 'video.mp4')
    output_dir = os.path.join(data_dir, 'outputs')
    output_video_path = os.path.join(output_dir, 'output_video.mp4')


    vp = VideoProcessor(video_path)
    ad = ActivityDetector()
    anom_detector = AnomalyDetector()
    summary = SummaryGenerator()

    # Configuração do writer de vídeo
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = vp.cap.get(cv2.CAP_PROP_FPS)
    width = int(vp.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vp.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    current_activity = None
    try:
        while True:
            frame = vp.get_next_frame()
            if frame is None:
                logging.info("Fim do vídeo alcançado ou falha na leitura do frame.")
                break

            logging.info(f"Processando frame {vp.frame_number} de {vp.total_frames}")

            # Incrementar o contador total de frames no resumo
            summary.total_frames += 1

            # Envolver o processamento de cada frame em um bloco try-except
            try:
                # Detecção de faces
                face_locations = detect_faces(frame)
                frame = mark_faces(frame, face_locations)

                # Inicializar variáveis para emoções e atividades
                emotion_detected = False
                activity_detected = False

                # Análise de emoções
                if face_locations:
                    emotions = analyze_emotions(frame, face_locations)
                    frame = annotate_emotions(frame, emotions)
                    if emotions:
                        for emotion_data in emotions:
                            emotion = emotion_data['emotion']
                            summary.update_emotion(emotion)
                            emotion_detected = True

                # Detecção de atividades
                if ad.add_frame(frame):
                    current_activity = ad.detect_activity()
                    if current_activity:
                        summary.update_activity(current_activity)
                        activity_detected = True
                # Exibir a atividade detectada no frame, na parte inferior
                cv2.putText(
                    frame,
                    f'Atividade: {current_activity}',
                    (10, frame.shape[0] - 10),  # Posição no rodapé
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 0),  # Cor do texto
                    2
                )

                # Detecção de anomalias
                anomaly_detected = False
                if face_locations:
                    for top, right, bottom, left in face_locations:
                        face_image = frame[top:bottom, left:right]
                        if anom_detector.detect_anomaly(face_image):
                            cv2.putText(frame, 'Careta detectada', (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                            anomaly_detected = True
                            summary.update_anomaly()
                            break  # Considera apenas uma anomalia por frame

                # registrar frames sem detecções
                if not (emotion_detected or activity_detected or anomaly_detected):
                    logging.debug("Nenhuma emoção, atividade ou anomalia detectada neste frame.")

            except Exception as e:
                logging.error(f"Erro ao processar o frame {vp.frame_number}: {e}")
                # continue

            # Escrever o frame processado no vídeo de saída
            out.write(frame)

            # Mostrar o frame processado em uma janela
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Após o loop, podemos finalizar o resumo se necessário
        summary.finalize()

    except Exception as e:
        logging.error(f"Exceção não tratada no programa: {e}")
    finally:
        # Liberação dos recursos
        vp.release()
        out.release()
        cv2.destroyAllWindows()
        anom_detector.close()  # Fecha o recurso do MediaPipe no AnomalyDetector

        # Salvar o resumo
        summary.save_summary()
        logging.info("Processamento concluído.")

if __name__ == '__main__':
    main()