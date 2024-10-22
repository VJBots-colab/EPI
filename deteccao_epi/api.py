from contextlib import asynccontextmanager
import os
import secrets
import logging
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status, Request
from fastapi.responses import JSONResponse
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
    # Código de inicialização
    FastAPICache.init(InMemoryBackend())
    yield
    # Código de encerramento (se necessário)

app = FastAPI(
    title="API de Detecção de EPIs",
    description="Esta API detecta Equipamentos de Proteção Individual (EPIs) em imagens.",
    version="1.0.0",
    lifespan=lifespan
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

security = HTTPBasic()

# Defina o caminho relativo do modelo
MODEL_PATH = os.path.join('runs', 'detect', 'epi_detector3', 'weights', 'best.pt')

# Obtenha o caminho absoluto
ABSOLUTE_MODEL_PATH = os.path.abspath(MODEL_PATH)

# Tente carregar o modelo usando YOLO
try:
    model = YOLO(ABSOLUTE_MODEL_PATH)
except Exception as e:
    print(f"Erro ao carregar o modelo com YOLO: {e}")
    # Se falhar, tente carregar com torch.load
    try:
        model = torch.load(ABSOLUTE_MODEL_PATH, map_location=torch.device('cpu'))
        if isinstance(model, dict):
            # Se o modelo for um dicionário, pode ser necessário extrair o modelo real
            model = model.get('model', model)
        model.eval()  # Coloque o modelo em modo de avaliação
    except Exception as e:
        print(f"Erro ao carregar o modelo com torch.load: {e}")
        raise

# Lista de classes (ajuste conforme necessário)
CLASSES = ['capacete', 'oculos', 'bota', 'mascara', 'luvas']

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

@app.post("/detect_epi", summary="Detectar EPIs em uma imagem")
@cache(expire=60)
async def detect_epi(file: UploadFile = File(...), username: str = Depends(get_current_username)):
    """
    Detecta Equipamentos de Proteção Individual (EPIs) em uma imagem enviada.

    - **file**: Arquivo de imagem (PNG, JPG, JPEG) contendo a cena a ser analisada.

    Retorna um dicionário com as probabilidades de detecção para cada categoria de EPI.
    """
    try:
        validate_image(file)
        if not file:
            raise HTTPException(status_code=400, detail="No file part")
        if file.filename == '':
            raise HTTPException(status_code=400, detail="No selected file")
        if not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail="File type not allowed")

        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File size exceeds maximum limit")

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        with open(filepath, "wb") as buffer:
            buffer.write(contents)

        logger.info(f"Processing file: {filename}")
        results = detect_epi_in_image(filepath)
        
        return results

    except HTTPException as e:
        logger.error(f"HTTP error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health", summary="Verificar status do serviço")
async def health_check():
    """
    Verifica o status do serviço de detecção de EPIs.
    """
    return {"status": "ok", "model_loaded": os.path.exists(ABSOLUTE_MODEL_PATH)}

@app.post("/detect_epi_multiple")
async def detect_epi_multiple(files: List[UploadFile] = File(...), username: str = Depends(get_current_username)):
    results = []
    for file in files:
        # Processe cada arquivo como antes
        result = await process_single_file(file)
        results.append(result)
    return results

async def process_single_file(file: UploadFile):
    # ... (código para processar um único arquivo)
    return await detect_epi(file)

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
    results = model(img)

    # Processe os resultados
    detections = []
    for result in results:
        for *box, conf, cls in result.boxes.data.tolist():
            detections.append({
                'class': CLASSES[int(cls)],
                'confidence': float(conf),
                'bbox': [float(coord) for coord in box]
            })

    logger.info(f"Resultados da detecção: {detections}")
    return JSONResponse(content={'detections': detections})

def is_valid_image(file_contents):
    try:
        Image.open(io.BytesIO(file_contents))
        return True
    except IOError:
        return False

def predict(image):
    if isinstance(model, YOLO):
        results = model(image)
        # Processe os resultados conforme necessário
        return results
    else:
        # Se não for um modelo YOLO, ajuste conforme necessário
        with torch.no_grad():
            output = model(image)
        # Processe o output conforme necessário
        return output

async def get_file_size(file: UploadFile):
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    return size

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
