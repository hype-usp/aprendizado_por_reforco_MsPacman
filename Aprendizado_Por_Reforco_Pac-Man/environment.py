import gym
from config import Config

# Cria o environment do pac-man
def create_environment():
    env = gym.make("MsPacman-v4", render_mode=Config.RENDER_MODE)
    return env