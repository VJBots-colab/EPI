import cv2
import torch
from ultralytics import YOLO
from config import EPI_CATEGORIES, MODEL_SAVE_PATH
import logging

logger = logging.getLogger(__name__)

def load_epi_model():
    """Carregamento do modelo treinado."""
    return YOLO(MODEL_SAVE_PATH)

def detect_epi(image):
    try:
        model = load_epi_model()
        
        results = model(image)
        
        detections = {}
        for r in results:
            for c in r.boxes.cls:
                category = EPI_CATEGORIES[int(c)]
                if category not in detections:
                    detections[category] = 1
                else:
                    detections[category] += 1
        
        return detections
    except Exception as e:
        logger.error(f"Erro na detecção de EPI: {str(e)}")
        raise

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
