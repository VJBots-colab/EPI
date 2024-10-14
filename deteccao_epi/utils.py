import cv2
import numpy as np

from deteccao_epi.config import IMAGE_SIZE, EPI_CATEGORIES

def preprocess_image(image_path):
    """Carrega e pré-processa uma imagem para o modelo."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converter para RGB
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype(np.float32) / 255.0  # Normalização
    return img

def load_and_preprocess_batch(image_paths):
    """Carrega e pré-processa um lote de imagens."""
    return np.array([preprocess_image(path) for path in image_paths])

def create_label(category, all_categories):
    """Cria um vetor de rótulos one-hot."""
    return [1 if cat == category else 0 for cat in all_categories]

def visualize_results(image, predictions):
    """Visualiza os resultados da detecção de EPIs."""
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for i, (category, prob) in enumerate(predictions.items()):
        text = f"{category}: {prob:.2f}"
        cv2.putText(img, text, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return img
