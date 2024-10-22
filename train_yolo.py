import os
from ultralytics import YOLO

# Configurações
DATA_DIR = os.path.join(os.getcwd(), 'data')
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 16

# Crie um arquivo YAML para configuração do dataset
def create_dataset_yaml():
    yaml_content = f"""
path: {DATA_DIR}
train: images/train
val: images/valid  # Note a mudança de 'val' para 'valid'
test: images/test

nc: 5
names: ['capacete', 'oculos', 'bota', 'mascara', 'luvas']
    """
    
    with open('dataset.yaml', 'w') as f:
        f.write(yaml_content)

# Treine o modelo
def train_yolo():
    create_dataset_yaml()  # Cria o arquivo YAML antes de treinar
    
    model = YOLO('yolov5s.pt')  # Carrega o modelo YOLOv5s pré-treinado
    
    results = model.train(
        data='dataset.yaml',
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name='epi_detector'
    )
    
    # Crie o diretório 'models' se ele não existir
    os.makedirs('models', exist_ok=True)
    
    # Salve o modelo treinado
    model.save('models/epi_detector.pt')

if __name__ == '__main__':
    train_yolo()
