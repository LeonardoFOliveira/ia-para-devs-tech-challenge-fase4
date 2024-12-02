# FIAP - Tech Challange - fase 4

## O PROBLEMA 
O Tech Challenge desta fase será a criação de uma aplicação que utilize 
análise de vídeo. O seu projeto deve incorporar as técnicas de reconhecimento 
facial, análise de expressões emocionais em vídeos e detecção de atividades. 

## A PROPOSTA DO DESAFIO 
Você deverá criar uma aplicação a partir do vídeo que se encontra 
disponível na plataforma do aluno, e que execute as seguintes tarefas: 
1. Reconhecimento facial: Identifique e marque os rostos presentes no 
vídeo. 
2. Análise de expressões emocionais: Analise as expressões 
emocionais dos rostos identificados. 
3. Detecção de atividades: Detecte e categorize as atividades sendo 
realizadas no vídeo. 
4. Geração de resumo: Crie um resumo automático das principais 
atividades e emoções detectadas no vídeo.   

## O QUE ESPERAMOS COMO ENTREGÁVEL? 
1. Código Fonte: todo o código fonte da aplicação deve ser entregue em 
um repositório Git, incluindo um arquivo README com instruções 
claras de como executar o projeto. 
2. Relatório: o resumo obtido automaticamente com as principais 
atividades e emoções detectadas no vídeo. Nesse momento 
esperando que o relatório inclua: 
- Total de frames analisados. 
- Número de anomalias detectadas.   
_Observação:  movimento anômalo não segue o padrão geral de atividades 
(como gestos bruscos ou comportamentos atípicos) esses são classificados 
como anômalos._ 
3. Demonstração em Vídeo: um vídeo demonstrando a aplicação em 
funcionamento, evidenciando cada uma das funcionalidades 
implementadas.

## Dataset

Vídeo a ser utilizado: https://drive.google.com/drive/folders/1sJdcVpF0fuAU1vAKoxKBeYeIUuMC9rQb

# Projeto desenvolvido - Análise de Vídeo com Reconhecimento Facial, Emoções e Detecção de Atividades

## Descrição

Este projeto realiza a análise de um vídeo, identificando e marcando rostos, analisando expressões emocionais, detectando atividades e identificando "caretas" como comportamentos anômalos. Ao final, gera um resumo automático com as principais atividades e emoções detectadas.

## Pré-requisitos

- Python 3.x
- Bibliotecas listadas em `requirements.txt`

## Instalação

1. Clone o repositório:

```bash
git clone https://github.com/LeonardoFOliveira/ia-para-devs-tech-challenge-fase4
cd ia-para-devs-tech-challenge-fase4
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Como Executar

### Treinamento do Modelo de Anomalia
Antes de executar o projeto, é necessário treinar o modelo de detecção de anomalias para identificar "caretas". Siga as instruções:
1. Certifique-se de que o vídeo de treinamento (`video_normal.mp4`) está na pasta `data/` do projeto. Este vídeo deve conter expressões faciais normais, sem "caretas".

2. Navegue até a pasta de treinamento:
```bash
cd training
```

3. Execute o script de treinamento:
```bash
python train_anomaly_detector.py
```

4. O modelo treinado será salvo na pasta `models/` como `anomaly_detector_model.pkl`.

### Executar aplicação principal
1. Certifique-se de que o vídeo a ser analisado `(video.mp4)` está na pasta `data` do projeto.

2. Execute o script principal:
```bash
python main.py
```

3. O processamento pode levar algum tempo dependendo do tamanho do vídeo.

4. Ao final, serão gerados em `data/outputs`:
- `output_video.mp4:` Vídeo com as detecções e anotações.
- `relatorio.txt:` Resumo da análise realizada.

## Estrutura dos Módulos
- `main.py`: Script principal que inicia a aplicação.
- `video_processor.py`: Classe para carregar e processar o vídeo.
- `face_recognition_module.py`: Funções para detecção e marcação de faces.
- `emotion_analysis_module.py`: Funções para análise e anotação de emoções.
- `activity_detection_module.py`: Classe para detecção de atividades.
- `anomaly_detection_module.py`: Classe para detecção de anomalias ("caretas").
- `summary_generator.py`: Classe para gerar o resumo automático.
- `training/train_anomaly_detector.py`: Script que treina o modelo de anomalia consideradas por "caretas".

