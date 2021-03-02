#  Drakkar-Software OctoBot-IA-Tentacles
#  Copyright (c) Drakkar-Software, All rights reserved.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 3.0 of the License, or (at your option) any later version.
#
#  This library is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public
#  License along with this library.
import os
import random
import time
from collections import deque

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam


# references:
# https://arxiv.org/pdf/1802.09477.pdf
# https://arxiv.org/pdf/1509.06461.pdf
# https://papers.nips.cc/paper/3964-double-q-learning.pdf
# https://github.com/Albert-Z-Guo/Deep-Reinforcement-Stock-Trading
class DDQNAgent:
    MODEL_NAME = "test"
    MODELS_PATH = "saved_models"
    LOGS_DIR = "/tmp/tensorboard"
    WINDOW_SIZE = 5

    def __init__(self, is_training):
        self.is_training = is_training
        self.window_size = self.WINDOW_SIZE

        self.model_type = 'DQN'
        self.state_dim = self.window_size + 3
        self.action_dim = 3  # hold, buy, sell
        self.memory = deque(maxlen=100)
        self.buffer_size = 10

        self.tau = 0.0001
        self.gamma = 0.95
        self.epsilon = 1.0  # initial exploration rate
        self.epsilon_min = 0.01  # minimum exploration rate
        self.epsilon_decay = 0.995  # decrease exploration rate as the agent becomes good at trading

        self.model_path = os.path.join(self.MODELS_PATH, self.MODEL_NAME + ".h5")
        self.model = load_model(self.model_path) if not self.is_training else self.model()
        self.model_target = load_model(
            os.path.join(self.MODELS_PATH, self.MODEL_NAME + "_target.h5")) if not self.is_training else self.model
        self.model_target.set_weights(self.model.get_weights())  # hard copy model parameters to target model parameters

        self.tensorboard = TensorBoard(log_dir=os.path.join(os.path.basename(self.model_path),
                                                            self.LOGS_DIR, str(time.time())),
                                       histogram_freq=0,
                                       profile_batch=0)
        self.tensorboard.set_model(self.model)

    def update_model_target(self):
        model_weights = self.model.get_weights()
        model_target_weights = self.model_target.get_weights()
        for i in range(len(model_weights)):
            model_target_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * model_target_weights[i]
        self.model_target.set_weights(model_target_weights)

    def model(self):
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_dim, activation='relu'))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=8, activation='relu'))
        model.add(Dense(self.action_dim, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model

    def reset(self):
        self.epsilon = 1.0

    def remember(self, state, actions, reward, next_state, done):
        self.memory.append((state, actions, reward, next_state, done))

    def act(self, state):
        if not not self.is_training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        options = self.model.predict(state)
        return np.argmax(options[0])

    def experience_replay(self):
        # sample random buffer_size long memory
        mini_batch = random.sample(self.memory, self.buffer_size)

        for state, actions, reward, next_state, done in mini_batch:
            Q_expected = reward + (1 - done) * self.gamma * np.amax(self.model_target.predict(next_state)[0])

            next_actions = self.model.predict(state)
            next_actions[0][np.argmax(actions)] = Q_expected

            history = self.model.fit(state, next_actions, epochs=1, verbose=0)
            self.update_model_target()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return history.history['loss'][0]

    def save_model(self, period):
        self.model.save(os.path.join(self.MODELS_PATH, self.MODEL_NAME + 'DQN_ep' + str(period) + '.h5'))
