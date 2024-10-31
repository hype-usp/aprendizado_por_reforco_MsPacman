import cv2
import numpy as np
from config import Config

def preprocess_observation(observation):
    # Converte para escala de cinza
    gray_observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    
    # Redimensiona para 84x84 pixels
    resized_observation = cv2.resize(gray_observation, (84, 84), interpolation=cv2.INTER_AREA)
    
    # Normaliza os valores de pixel para estarem entre 0 e 1
    normalized_observation = resized_observation / 255.0
    
    # Resultado final será shape (84, 84, 1)
    processed_observation = np.expand_dims(normalized_observation, axis=-1)
    
    return processed_observation