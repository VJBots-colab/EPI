import random

from typing import Dict

def detect_epi_in_image(image_path: str) -> Dict[str, float]:
    # Simula deteccao
    epi_categories = ['capacete', 'oculos', 'bota', 'mascara', 'luvas']
    return {category: random.uniform(0.5, 1.0) for category in epi_categories}