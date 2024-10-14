import os
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from config import EPI_CATEGORIES, IMAGE_SIZE, BATCH_SIZE, EPOCHS, MODEL_SAVE_PATH, TRAIN_DATA_DIR
from utils import load_and_preprocess_batch, create_label

def create_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    output = Dense(len(EPI_CATEGORIES), activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_data():
    images = []
    labels = []
    for category in EPI_CATEGORIES:
        category_path = os.path.join(TRAIN_DATA_DIR, category)
        for image_name in os.listdir(category_path):
            image_path = os.path.join(category_path, image_name)
            images.append(image_path)
            labels.append(create_label(category, EPI_CATEGORIES))
    
    X = load_and_preprocess_batch(images)
    y = np.array(labels)
    return X, y

def train_model():
    model = create_model()
    
    print("Carregando dados...")
    X_train, y_train = load_data()
    
    print("Iniciando treinamento...")
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)
    
    print("Salvando modelo...")
    model.save(MODEL_SAVE_PATH)
    
    print("Modelo treinado e salvo com sucesso!")

if __name__ == "__main__":
    train_model()
