from collections import Counter

class SummaryGenerator:
    def __init__(self):
        self.emotions_list = []
        self.activities_list = []
        self.anomalies_count = 0
        self.total_frames = 0

    def update(self, emotion, activity, anomaly_detected):
        self.emotions_list.append(emotion)
        self.activities_list.append(activity)
        if anomaly_detected:
            self.anomalies_count += 1
        self.total_frames += 1

    def generate_summary(self):
        dominant_emotions = Counter(self.emotions_list).most_common()
        dominant_activities = Counter(self.activities_list).most_common()

        summary = f"""Resumo da Análise do Vídeo:

- Total de frames analisados: {self.total_frames}
- Emoções predominantes:
"""
        for emotion, count in dominant_emotions:
            summary += f"  - {emotion}: {count} ocorrências\n"

        summary += "- Atividades predominantes:\n"
        for activity, count in dominant_activities:
            summary += f"  - {activity}: {count} ocorrências\n"

        summary += f"- Número de anomalias detectadas (caretas): {self.anomalies_count}\n"

        return summary

    def save_summary(self, filename='relatorio.txt'):
        summary = self.generate_summary()
        with open(filename, 'w') as f:
            f.write(summary)