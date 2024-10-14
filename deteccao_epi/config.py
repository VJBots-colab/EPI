import os

EPI_CATEGORIES = ['capacete', 'oculos', 'bota', 'mascara', 'luvas']

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

DATA_DIR = '../data'
RAW_DATA_DIR = f'{DATA_DIR}/raw'
PROCESSED_DATA_DIR = f'{DATA_DIR}/processed'
MODEL_SAVE_PATH = '../models/modelo_epi.h5'

TRAIN_DATA_DIR = f'{DATA_DIR}/treino'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 5 * 1024 * 1024
