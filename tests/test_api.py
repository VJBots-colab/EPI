import os
import io
import pytest
import base64
from dotenv import load_dotenv
from sklearn.metrics import precision_score, recall_score, f1_score

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
    assert "epi_model_loaded" in response.json()
    assert "fire_model_loaded" in response.json()
    assert response.json()["status"] == "ok"
    assert response.json()["epi_model_loaded"] is True
    assert response.json()["fire_model_loaded"] is True

def test_detect_fire_success():
    response = client.post("/detect_fire", files={"file": ("test.png", create_test_image(), "image/png")}, headers=get_auth_headers())
    assert response.status_code == 200
    assert "detections" in response.json()
    detections = response.json()["detections"]
    assert isinstance(detections, list)
    if detections:
        assert all(key in detections[0] for key in ['class', 'confidence', 'bbox'])
        assert detections[0]['class'] == 'fire'

def test_detect_all_success():
    response = client.post("/detect_all", files={"file": ("test.png", create_test_image(), "image/png")}, headers=get_auth_headers())
    assert response.status_code == 200
    assert "epi_detections" in response.json()
    assert "fire_detections" in response.json()
    epi_detections = response.json()["epi_detections"]
    fire_detections = response.json()["fire_detections"]
    assert isinstance(epi_detections, list)
    assert isinstance(fire_detections, list)
    if epi_detections:
        assert all(key in epi_detections[0] for key in ['class', 'confidence', 'bbox'])
        assert epi_detections[0]['class'] in ['capacete', 'oculos', 'bota', 'mascara', 'luvas']
    if fire_detections:
        assert all(key in fire_detections[0] for key in ['class', 'confidence', 'bbox'])
        assert fire_detections[0]['class'] == 'fire'

def test_with_annotated_data():
    annotated_data = [
        {"file": "tests/test_image.jpg", "expected": [{"class": "capacete", "confidence": 0.6}, {"class": "luvas", "confidence": 0.6}]},
    ]

    for data in annotated_data:
        with open(data["file"], "rb") as f:
            response = client.post("/detect", files={"file": (data["file"], f, "image/jpeg")}, headers=get_auth_headers())

        assert response.status_code == 200
        detections = response.json()["detections"]
        print("Detecções:", detections)  # Adicione esta linha para depuração
        for expected in data["expected"]:
            assert any(d["class"] == expected["class"] and d["confidence"] >= expected["confidence"] for d in detections)

def evaluate_model():
    y_true = []  # Classes verdadeiras
    y_pred = []  # Classes previstas

    annotated_data = [
        {"file": "tests/test_image.jpg", "expected": ["capacete", "luvas"]},
    ]

    for data in annotated_data:
        with open(data["file"], "rb") as f:
            response = client.post("/detect", files={"file": (data["file"], f, "image/jpeg")}, headers=get_auth_headers())
        
        detections = response.json()["detections"]
        predicted_classes = [d["class"] for d in detections]
        
        y_true.extend(data["expected"])
        y_pred.extend(predicted_classes)

    # Certifique-se de que y_true e y_pred têm o mesmo tamanho
    if len(y_true) != len(y_pred):
        print("Tamanhos diferentes: y_true =", len(y_true), "y_pred =", len(y_pred))
        return

    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"Precisão: {precision}, Recall: {recall}, F1-Score: {f1}")

# Comente a chamada direta para evitar execução fora do contexto de teste
# evaluate_model()

def test_edge_cases():
    edge_cases = [
        "tests/fire.jpg",
        # Adicione mais casos se necessário
    ]
    
    for case in edge_cases:
        with open(case, "rb") as f:
            response = client.post("/detect", files={"file": (case, f, "image/jpeg")}, headers=get_auth_headers())
        
        assert response.status_code == 200
        # Verifique as detecções conforme necessário

# Chame a função test_edge_cases() no final do seu arquivo de teste
test_edge_cases()
