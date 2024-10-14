import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from deteccao_epi.config import MODEL_SAVE_PATH, EPI_CATEGORIES
from deteccao_epi.utils import preprocess_image, visualize_results

def test_single_image(image_path):
    # Carregamento do modelo
    model = load_model(MODEL_SAVE_PATH)

    # Pré-processar a imagem
    img = preprocess_image(image_path)

    # Previsão
    prediction = model.predict(np.expand_dims(img, axis=0))[0]

    results = {cat: float(pred) for cat, pred in zip(EPI_CATEGORIES, prediction)}

    vis_img = visualize_results(image_path, results)

    plt.imshow(vis_img)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    test_image_path = "caminho/imagem/teste.jpg"
    test_single_image(test_image_path)