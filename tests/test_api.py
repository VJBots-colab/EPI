import os
import io
import pytest

from fastapi.testclient import TestClient
from deteccao_epi.api import app
from PIL import Image

client = TestClient(app)

def create_test_image():
    image = Image.new('RGB', (100, 100), color = 'red')
    img_byte_arr= io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

def test_detect_epi_success():
    response = client.post("/detect_epi", files={"file": ("test.png", create_test_image(), "image/png")})
    assert response.status_code == 200
    assert all(key in response.json() for key in ['capacete', 'oculos', 'bota', 'mascara', 'luvas'])

def test_detect_epi_no_file():
    response = client.post("/detect_epi")
    assert response.status_code == 422

def test_detect_epi_invalid_file_type():
    response = client.post("/detect_epi", files={"file": ("test.txt", b"test content", "text/plain")})
    assert response.status_code == 400
    assert "File type not allowed" in response.json()["detail"]

def test_detect_epi_large_file():
    large_image = create_test_image() * 1000000  # Create a large file
    response = client.post("/detect_epi", files={"file": ("large.png", large_image, "image/png")})
    assert response.status_code == 400
    assert "File size exceeds maximum limit" in response.json()["detail"]