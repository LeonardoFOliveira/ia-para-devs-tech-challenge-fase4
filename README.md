# Análise de Vídeo com Reconhecimento Facial, Emoções e Detecção de Atividades

## Descrição

Este projeto realiza a análise de um vídeo, identificando e marcando rostos, analisando expressões emocionais, detectando atividades e identificando "caretas" como comportamentos anômalos. Ao final, gera um resumo automático com as principais atividades e emoções detectadas.

## Pré-requisitos

- Python 3.x
- Bibliotecas listadas em `requirements.txt`

## Instalação

1. Clone o repositório:

```bash
git clone https://github.com/seu_usuario/seu_repositorio.git
cd seu_repositorio
```

2. Instale as dependências:
  - Instale primeiramente o `face_recognition` sem chace:
```bash
pip install --no-cache-dir face_recognition
```
  - Instale as demais depedências pelo `requirements.txt`:
```bash
pip install -r requirements.txt
```
_**Nota:** Pode ser necessário instalar `dlib` separadamente devido a requisitos de compilação. Consulte a documentação oficial do `dlib` para instruções detalhadas._
Pode ser necessário por exemplo em windows: 
- _Baixar o CMake em seu sistema (https://cmake.org/download/)_
- _Baixar o Visual Studio Installer (https://visualstudio.microsoft.com/pt-br/downloads/)_
  - _Baixar o componente individual de `Ferramentas do CMake do C++ para Windows`_
  - _Baixar o componente individual de `SDK do Windows` de acordo com a versão do seu Windows._


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

