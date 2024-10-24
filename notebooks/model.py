from nb_utils import set_root
PROJECT_DIR = set_root(2)


import numpy as np
import pandas as pd
import supervision as sv

import ultralytics
from ultralytics import YOLO
ultralytics.checks()


path_data = PROJECT_DIR / "data"
path_intermediate = path_data / "02_intermediate"

file_path_tracker = path_intermediate / "tracker_all.csv"


# Load a model
model = YOLO("yolov8s.yaml")  # build a new model from scratch
model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)

# Use the model
results = model.train(data="data.yaml", epochs=3)  # train the model
results = model.val()  # evaluate model performance on the validation set



# Inicializando o modelo, o rastreador e os anotadores
model = YOLO("runs/detect/train/weights/best.pt")
tracker = sv.ByteTrack()
annotator = sv.RoundBoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

# Criando um DataFrame vazio para armazenar as detecções
df_detections = pd.DataFrame(columns=["tracker_id", "class_id", "x_min", "y_min", "x_max", "y_max"])

# Função de callback para processar cada frame
def callback(frame: np.ndarray, _: int) -> np.ndarray:
    global df_detections

    # Fazendo a predição no frame atual
    results = model(frame)[0]

    # Convertendo resultados para o formato de detecção
    detections = sv.Detections.from_ultralytics(results)

    # Atualizando os rastreamentos com o ByteTrack
    detections = tracker.update_with_detections(detections)
    
    # Extraindo informações das detecções para o DataFrame
    new_data = []
    for i, (class_id, tracker_id, box) in enumerate(zip(detections.class_id, detections.tracker_id, detections.xyxy)):
        # Extraindo coordenadas da bounding box
        x_min, y_min, x_max, y_max = box
        
        # Adicionando as informações à lista de novas linhas
        new_data.append({
            "tracker_id": tracker_id,
            "class_id": class_id,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max
        })

    # Concatenando novas linhas ao DataFrame
    df_detections = pd.concat([df_detections, pd.DataFrame(new_data)], ignore_index=True)

    # Criando rótulos para cada detecção
    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ]

    # Anotando o frame com as caixas de detecção, rótulos e rastros
    annotated_frame = annotator.annotate(frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
    return trace_annotator.annotate(annotated_frame, detections=detections)

# Processando o vídeo e salvando o resultado
sv.process_video(
    source_path="11.2.mp4",
    target_path="resultIDTraceRo.mp4",
    callback=callback
)

# Exibindo as primeiras linhas do DataFrame ao final do processamento
print(df_detections.head())
