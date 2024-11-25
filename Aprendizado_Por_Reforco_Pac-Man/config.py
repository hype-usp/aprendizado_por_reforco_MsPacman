class Config:
    # Configurações do Ambiente
    RENDER_MODE = "human"
    ENV_NAME = "MsPacman-v4"

    # Hiperparâmetros do Agente
    GAMMA = 0.99  # Fator de desconto
    EPSILON_INITIAL = 1.0  # Probabilidade inicial de exploração
    EPSILON_MIN = 0.1  # Limite mínimo de epsilon
    EPSILON_DECAY = 0.995  # Taxa de decaimento de epsilon
    LEARNING_RATE = 0.001  # Taxa de aprendizado
