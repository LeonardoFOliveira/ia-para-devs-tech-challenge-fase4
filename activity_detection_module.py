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
        ])
        self.sequence_length = 16
        self.frames = []

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
        # Aqui, pode-se usar a função torch.argmax e um dicionário de labels
        _, predicted = torch.max(outputs.data, 1)
        activity_label = self._get_label(predicted.item())
        return activity_label

    def _get_label(self, index):
        # Dicionário de mapeamento de índices para atividades
        # Substitua pelos labels corretos do seu modelo
        labels = {
            0: 'aplaudindo',
            1: 'andando',
            2: 'correndo',
            # Adicione outros labels conforme necessário
        }
        return labels.get(index, 'atividade_desconhecida')