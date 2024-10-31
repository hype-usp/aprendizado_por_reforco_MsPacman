from environment import create_environment
from preprocess import preprocess_observation
import cv2
import numpy as np

def main():
    # Cria o ambiente
    env = create_environment()
    
    # Reinicializa o ambiente e obtém a primeira observação
    observation, info = env.reset()
    
    # Inicializa variáveis para o loop do jogo
    done = False
    total_reward = 0

    # Loop do jogo até o episódio terminar
    while not done:
        # Preprocessa a observação atual
        processed_observation = preprocess_observation(observation)

        # Exibe a observação original e preprocessada
        cv2.imshow("Original Observation", observation)
        processed_for_display = (processed_observation * 255).astype('uint8')
        cv2.imshow("Processed Observation", processed_for_display)

        # Aguarda um breve momento para permitir a visualização
        cv2.waitKey(1)

        # Escolhe uma ação aleatória
        action = env.action_space.sample()

        # Executa a ação no ambiente
        observation, reward, terminated, truncated, info = env.step(action)

        # Atualiza a pontuação total e verifica se o jogo terminou
        total_reward += reward
        done = terminated or truncated
    
    # Exibe a pontuação total do jogo
    print(f"Pontuação do episódio: {total_reward}")
    
    # Fecha as janelas e o ambiente
    cv2.destroyAllWindows()
    env.close()

if __name__ == "__main__":
    main()