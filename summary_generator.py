import os

class SummaryGenerator:
    def __init__(self):
        self.total_frames = 0
        self.emotion_counts = {}
        self.activity_counts = {}
        self.anomaly_count = 0

    def update_emotion(self, emotion):
        # Atualizar contagem de emoções
        if emotion not in self.emotion_counts:
            self.emotion_counts[emotion] = 0
        self.emotion_counts[emotion] += 1

    def update_activity(self, activity):
        # Atualizar contagem de atividades
        if activity not in self.activity_counts:
            self.activity_counts[activity] = 0
        self.activity_counts[activity] += 1

    def update_anomaly(self):
        # Atualizar contagem de anomalias
        self.anomaly_count += 1

    def finalize(self):
        pass

    def save_summary(self):
        project_root = os.path.dirname(os.path.abspath(__file__))
        report_path = os.path.join(project_root, 'data', 'outputs', 'relatorio.txt')
        with open(report_path, 'w',  encoding='utf-8') as f:
            f.write(f"Total de frames processados: {self.total_frames}\n\n")

            if self.emotion_counts:
                f.write("Emoções detectadas:\n")
                for emotion, count in self.emotion_counts.items():
                    percentage = (count / self.total_frames) * 100
                    f.write(f"- {emotion}: {count} frames ({percentage:.2f}%)\n")
                f.write("\n")
            else:
                f.write("Nenhuma emoção detectada.\n\n")

            if self.activity_counts:
                f.write("Atividades detectadas:\n")
                for activity, count in self.activity_counts.items():
                    percentage = (count / self.total_frames) * 100
                    f.write(f"- {activity}: {count} frames ({percentage:.2f}%)\n")
                f.write("\n")
            else:
                f.write("Nenhuma atividade detectada.\n\n")

            f.write(f"Número de anomalias detectadas: {self.anomaly_count}\n")