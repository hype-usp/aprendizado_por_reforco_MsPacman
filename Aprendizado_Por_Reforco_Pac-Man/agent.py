import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
from config import GAMMA, EPSILON_INITIAL, EPSILON_MIN, EPSILON_DECAY, LEARNING_RATE

class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size

        # Hiperparâmetros
        self.gamma = GAMMA
        self.epsilon = EPSILON_INITIAL
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE

        # Inicialização da rede neural
        self.model = self.build_model()

    def build_model(self):
        # Construção da Rede Neural para predição dos valores Q
        model = Sequential()
        model.add(Flatten(input_shape=self.state_shape))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def choose_action(self, state):
        # Implementação da política ε-greedy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
            return np.argmax(q_values[0])

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
