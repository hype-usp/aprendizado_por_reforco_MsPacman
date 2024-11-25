import gym
from config import RENDER_MODE, ENV_NAME

# Cria o environment do pac-man
def create_environment():
    env = gym.make(ENV_NAME, render_mode=RENDER_MODE)
    return env

# Reinicia o ambiente e obtém a primeira observação
def reset_environment(env):
    observation, info = env.reset()
    return observation, info

# Executa uma ação no ambiente
def step_environment(env, action):
    return env.step(action)
