import cv2
import numpy as np

from tensorflow.keras.models import load_model
from config import EPI_CATEGORIES, MODEL_SAVE_PATH, IMAGE_SIZE
from utils import preprocess_image

def load_epi_model():
    """Carregamento do modelo treinado."""
    return load_model(MODEL_SAVE_PATH)

def detect_epi(image, model):
    """Detecta EPIs em uma imagem."""
    processed_image = preprocess_image(image)
    prediction = model.predict(np.expand_dims(processed_image, axis=0))[0]
    return {cat: float(pred) for cat, pred in zip(EPI_CATEGORIES, prediction)}

def main():
    # model = load_epi_model()

    # captura de video
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        cv2.imshow('EPI Detection', frame)
        
        if cv2.waitKey('1') & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()