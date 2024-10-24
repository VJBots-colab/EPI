from contextlib import asynccontextmanager
import os
import secrets
import logging
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status, Request
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from werkzeug.utils import secure_filename
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache
from typing import List
from PIL import Image
from dotenv import load_dotenv
import io
import torch
from ultralytics import YOLO

from deteccao_epi.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE, UPLOAD_FOLDER
from deteccao_epi.detector import detect_epi_in_image

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    FastAPICache.init(InMemoryBackend())
    yield

app = FastAPI(lifespan=lifespan)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

security = HTTPBasic()

# Carregue os modelos
EPI_MODEL_PATH = 'runs/detect/epi_detector3/weights/best.pt'
FIRE_MODEL_PATH = 'runs/detect/epi_detector/weights/best.pt'

epi_model = YOLO(EPI_MODEL_PATH)
fire_model = YOLO(FIRE_MODEL_PATH)

# Defina as classes para cada modelo
EPI_CLASSES = ['capacete', 'oculos', 'bota', 'mascara', 'luvas']
FIRE_CLASSES = ['fire']

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image(file: UploadFile):
    try:
        img = Image.open(file.file)
        img.verify()
    except:
        raise HTTPException(status_code=400, detail="Invalid image file")
    finally:
        file.file.seek(0)

def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = os.getenv("API_USERNAME")
    correct_password = os.getenv("API_PASSWORD")
    is_correct_username = secrets.compare_digest(credentials.username, correct_username)
    is_correct_password = secrets.compare_digest(credentials.password, correct_password)
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

@app.get("/health", summary="Verificar status do serviço")
async def health_check():
    """
    Verifica o status do serviço de detecção de EPIs e focos de incêndio.
    """
    return {
        "status": "ok", 
        "epi_model_loaded": os.path.exists(EPI_MODEL_PATH),
        "fire_model_loaded": os.path.exists(FIRE_MODEL_PATH)
    }

@app.post("/detect")
@cache(expire=60)
async def detect_epi_endpoint(request: Request, file: UploadFile = File(...), username: str = Depends(get_current_username)):
    # Verifique o tamanho do arquivo
    file_size = await get_file_size(file)
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Arquivo muito grande. O tamanho máximo permitido é 10 MB.")

    logger.info(f"Recebida solicitação para detectar EPI na imagem: {file.filename}")
    contents = await file.read()

    if not is_valid_image(contents):
        logger.error(f"Arquivo inválido: {file.filename}")
        raise HTTPException(status_code=400, detail="Arquivo inválido. Por favor, envie uma imagem válida.")

    img = Image.open(io.BytesIO(contents))

    # Realize a detecção usando o modelo YOLOv5
    results = epi_model(img)  # Use epi_model em vez de model

    # Processe os resultados
    detections = []
    for result in results:
        for *box, conf, cls in result.boxes.data.tolist():
            detections.append({
                'class': EPI_CLASSES[int(cls)],
                'confidence': float(conf),
                'bbox': [float(coord) for coord in box]
            })

    logger.info(f"Resultados da detecção: {detections}")
    return {'detections': detections}

@app.post("/detect_fire", summary="Detectar focos de incêndio em uma imagem")
@cache(expire=60)
async def detect_fire_endpoint(file: UploadFile = File(...), username: str = Depends(get_current_username)):
    """
    Detecta focos de incêndio industrial em umma imagem enviada.
    """
    return await process_image(file, fire_model, FIRE_CLASSES)

@app.post("/detect_all", summary="Detectar EPIs e focos de incêndio em uma imagem.")
@cache(expire=60)
async def detect_all_endpoints(file: UploadFile = File(...), username: str = Depends(get_current_username)):
    """
    Detecta EPIs e focos de incêndio industrial em uma imagem enviada
    """
    try:
        epi_results = await process_image(file, epi_model, EPI_CLASSES)
        # Rewind the file after the first read
        await file.seek(0)
        fire_results = await process_image(file, fire_model, FIRE_CLASSES)
        return {
            "epi_detections": epi_results["detections"],
            "fire_detections": fire_results["detections"]
        }
    except Exception as e:
        logger.error(f"Erro ao processar imagem: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Erro ao processar imagem: {str(e)}")

def is_valid_image(file_contents):
    try:
        Image.open(io.BytesIO(file_contents))
        return True
    except IOError:
        return False

def predict(image, model):
    if isinstance(model, YOLO):
        results = model(image)
        return results
    else:
        with torch.no_grad():
            output = model(image)
        return output

async def get_file_size(file: UploadFile):
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    return size

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="API de Detecção de EPIs e Focos de Incêndio",
        version="2.0.0",
        description="Esta API permite a detecção de Equipamentos de Proteção Individual (EPIs) e focos de incêndio industrial em imagens.",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

@app.post("/detect_fire", summary="Detectar focos de incêndio em uma imagem")
async def detect_fire_endpoint(request: Request, file: UploadFile = File(...), username: str = Depends(get_current_username)):
    """
    Detecta focos de incêndio industrial em uma imagem enviada.

    - **file**: Arquivo de imagem (PNG, JPG, JPEG) contendo a cena a ser analisada.
    - Retorna uma lista de detecções, cada uma contendo a classe (fogo), confiança e coordenadas do bounding box.
    """
    # ... (código similar ao endpoint de detecção de EPIs, mas usando o novo modelo)

async def process_image(file: UploadFile, model, classes):
    contents = await file.read()
    if not is_valid_image(contents):
        raise HTTPException(status_code=400, detail="Arquivo inválido. Por favor, envie uma imagem válida.")
    
    img = Image.open(io.BytesIO(contents))
    results = model(img)
    
    detections = []
    for result in results:
        for *box, conf, cls in result.boxes.data.tolist():
            if int(cls) < len(classes):
                detection = {
                    'class': classes[int(cls)],
                    'confidence': float(conf),
                    'bbox': [float(coord) for coord in box]
                }
                detections.append(detection)
                logger.info(f"Detecção: {detection}")
            else:
                logger.error(f"Classe inesperada: {cls}")
    
    return {"detections": detections}

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
