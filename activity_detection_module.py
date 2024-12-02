import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
from PIL import Image

class ActivityDetector:
    def __init__(self):
        # Carrega o modelo pré-treinado
        self.model = models.video.r3d_18(pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.43216, 0.394666, 0.37645],
                std=[0.22803, 0.22145, 0.216989]
            )
        ])
        self.sequence_length = 16
        self.frames = []
        # Carrega as labels do arquivo
        self.labels = self.load_labels('label_map.txt')

    def load_labels(self, filepath):
        labels = []
        with open(filepath, 'r') as f:
            for line in f:
                label = line.strip()
                labels.append(label)
        return labels

    def add_frame(self, frame):
        # Pré-processa o frame e adiciona à sequência
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        processed_frame = self.transform(image)
        self.frames.append(processed_frame)
        if len(self.frames) == self.sequence_length:
            return True
        return False

    def detect_activity(self):
        # Cria um tensor de entrada a partir dos frames
        input_tensor = torch.stack(self.frames).permute(1, 0, 2, 3).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(input_tensor)
        # Processa a saída para obter a atividade
        activity_label = self._process_outputs(outputs)
        self.frames = []
        return activity_label

    def _process_outputs(self, outputs):
        # Implementa a lógica para mapear a saída para um rótulo de atividade
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top_prob, top_catid = torch.topk(probabilities, 1)
        activity_label = self._get_label(top_catid.item())
        return activity_label

    def _get_label(self, index):
        if 0 <= index < len(self.labels):
            return self.labels[index]
        else:
            return 'atividade_desconhecida'