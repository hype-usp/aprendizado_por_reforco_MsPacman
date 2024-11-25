from environment import create_environment, reset_environment, step_environment
from preprocess import preprocess_observation
from agent import DQNAgent
import numpy as np
import cv2

def main():
    # Cria o ambiente
    env = create_environment()
    
    # Obtém informações sobre o estado e ações
    state_shape = env.observation_space.shape  # Dimensão do estado do ambiente
    action_size = env.action_space.n  # Número de ações possíveis

    # Cria o agente DQN
    agent = DQNAgent(state_shape, action_size)
    
    # Loop de Jogo para inicializar
    done = False
    total_reward = 0

    observation, info = reset_environment(env)
    
    while not done:
        # Preprocessa a observação atual
        processed_observation = preprocess_observation(observation)

        # Escolhe uma ação usando a política ε-greedy do agente
        action = agent.choose_action(processed_observation)

        # Executa a ação no ambiente
        observation, reward, terminated, truncated, info = step_environment(env, action)

        # Atualiza a pontuação total e verifica se o jogo terminou
        total_reward += reward
        done = terminated or truncated

        # Exibe a observação original e preprocessada
        cv2.imshow("Original Observation", observation)
        processed_for_display = (processed_observation * 255).astype('uint8')
        cv2.imshow("Processed Observation", processed_for_display)
        cv2.waitKey(1)

    # Exibe a pontuação total do jogo
    print(f"Pontuação do episódio: {total_reward}")
    
    # Fecha as janelas e o ambiente
    cv2.destroyAllWindows()
    env.close()

if __name__ == "__main__":
    main()
