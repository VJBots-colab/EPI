import os
import secrets
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from werkzeug.utils import secure_filename
from deteccao_epi.detector import detect_epi_in_image
from deteccao_epi.config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS, MAX_FILE_SIZE
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi_cache2 import FastAPICache
from fastapi_cache2.backends.inmemory import InMemoryBackend
from fastapi_cache2.decorator import cache
from typing import List
from PIL import Image

app = FastAPI(
    title="API de Detecção de EPIs",
    description="Esta API detecta Equipamentos de Proteção Individual (EPIs) em imagens.",
    version="1.0.0"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

security = HTTPBasic()

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.on_event("startup")
async def startup():
    FastAPICache.init(InMemoryBackend())

def validate_image(file: UploadFile):
    try:
        img = Image.open(file.file)
        img.verify()
    except:
        raise HTTPException(status_code=400, detail="Invalid image file")
    finally:
        file.file.seek(0)

@app.post("/detect_epi", summary="Detectar EPIs em uma imagem")
@cache(expire=60)
async def detect_epi(file: UploadFile = File(...), username: str = Depends(security.get_password)):
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
    return {"status": "ok"}

@app.post("/detect_epi_multiple")
async def detect_epi_multiple(files: List[UploadFile] = File(...), username: str = Depends(security.get_password)):
    results = []
    for file in files:
        # Processe cada arquivo como antes
        result = await process_single_file(file)
        results.append(result)
    return results

async def process_single_file(file: UploadFile):
    # ... (código para processar um único arquivo)
    return await detect_epi(file)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
