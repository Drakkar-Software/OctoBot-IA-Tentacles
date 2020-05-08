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
from collections import deque

import numpy as np
import time
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from tentacles.Evaluator.Util.keras.custom_tensorboard import CustomTensorBoard


class KerasAgent:
    LOGS_DIR = "/tmp/tensorboard"

    def __init__(self, state_size, is_training=False, model_path=""):
        self.state_size = state_size  # normalized previous days
        self.action_size = 3  # sit, buy, sell
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_path = model_path
        self.is_training = is_training

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        if self.is_training:
            disable_tensorflow_logs()
            self.tensorboard = CustomTensorBoard(log_dir=os.path.join(os.path.basename(self.model_path),
                                                                      self.LOGS_DIR, str(time.time())),
                                                 histogram_freq=0,
                                                 profile_batch=0)

        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
        else:
            # raise exception if not self.is_training
            self.model = load_model(self.model_path) if not self.is_training else self._model()

    def _model(self):
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.0005))

        return model

    def act(self, state):
        if self.is_training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        options = self.model.predict(state)
        return np.argmax(options[0])

    def exp_repay(self, batch_size):
        mini_batch = []
        memory_size = len(self.memory)
        for i in range(memory_size - batch_size + 1, memory_size):
            mini_batch.append(self.memory[i])

        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0,
                           callbacks=[self.tensorboard] if self.is_training else [])

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def disable_tensorflow_logs():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
