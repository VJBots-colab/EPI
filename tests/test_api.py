import os
import io
import pytest
import base64
from dotenv import load_dotenv

load_dotenv()

from fastapi.testclient import TestClient
from deteccao_epi.api import app
from PIL import Image

client = TestClient(app)

# Função auxiliar para autenticação
def get_auth_headers():
    username = os.getenv("API_USERNAME", "admin")
    password = os.getenv("API_PASSWORD", "password")
    return {
        "Authorization": f"Basic {base64.b64encode(f'{username}:{password}'.encode()).decode()}"
    }

def create_test_image():
    image = Image.new('RGB', (100, 100), color = 'red')
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

def test_detect_epi_success():
    response = client.post("/detect", files={"file": ("test.png", create_test_image(), "image/png")}, headers=get_auth_headers())
    assert response.status_code == 200
    assert "detections" in response.json()
    detections = response.json()["detections"]
    assert isinstance(detections, list)
    if detections:
        assert all(key in detections[0] for key in ['class', 'confidence', 'bbox'])

def test_detect_epi_no_file():
    response = client.post("/detect", headers=get_auth_headers())
    assert response.status_code == 422

def test_detect_epi_invalid_file_type():
    response = client.post("/detect", files={"file": ("test.txt", b"test content", "text/plain")}, headers=get_auth_headers())
    assert response.status_code == 400
    assert "Arquivo inválido" in response.json()["detail"]

def test_detect_epi_large_file():
    # Crie um arquivo que exceda o limite de 10 MB
    large_image = create_test_image() * 1000000  # Isso deve criar um arquivo maior que 10 MB
    response = client.post("/detect", files={"file": ("large.png", large_image, "image/png")}, headers=get_auth_headers())
    assert response.status_code == 413
    assert "Arquivo muito grande" in response.json()["detail"]

def test_detect_epi_with_real_image():
    test_image_path = os.path.join(os.path.dirname(__file__), "test_image.jpg")
    if not os.path.exists(test_image_path):
        pytest.skip("Test image not found")
    
    with open(test_image_path, "rb") as f:
        response = client.post("/detect", files={"file": ("test_image.jpg", f, "image/jpeg")}, headers=get_auth_headers())
    
    assert response.status_code == 200
    assert "detections" in response.json()
    detections = response.json()["detections"]
    assert isinstance(detections, list)
    if detections:
        assert all(key in detections[0] for key in ['class', 'confidence', 'bbox'])
        assert detections[0]['class'] in ['capacete', 'oculos', 'bota', 'mascara', 'luvas']

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "model_loaded" in response.json()
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True

# Remova ou comente o teste da função predict, já que ela não é mais importada diretamente
# def test_predict():
#     # Implemente seu teste aqui
#     pass
