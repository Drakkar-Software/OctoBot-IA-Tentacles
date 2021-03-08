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
import json
import logging
import os
import pickle
import random
import time
from collections import deque

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model, clone_model
from keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

from tentacles.Trading.Mode.q_trading_mode.utils import huber_loss


class QAgent:
    LOGS_DIR = "/tmp/tensorboard"

    def __init__(self, is_training, state_size, action_size, output_size, model_path, model_name):
        self.is_training = is_training
        self.model_name = model_name
        self.model_path = model_path

        # agent config
        self.state_size = state_size
        self.action_size = action_size
        self.output_size = output_size
        self.model_name = model_name
        self.memory = deque(maxlen=10000)

        # model config
        self.model_name = model_name
        self.gamma = 0.95  # affinity for long term reward
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.loss = huber_loss
        self.custom_objects = {"huber_loss": huber_loss}  # important for loading the model from memory
        self.optimizer = Adam(lr=self.learning_rate)

        if not self.is_training and self.model_name is not None:
            self.model = self.load()
        else:
            try:
                self.model = self.load()
            except OSError as e:
                logging.getLogger().error(f"Impossible to load existing model : {e}")
            self.model = self._model()

        # target network
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

        self.tensorboard = TensorBoard(log_dir=os.path.join(os.path.basename(self.model_path),
                                                            self.LOGS_DIR, str(time.time())),
                                       histogram_freq=0,
                                       profile_batch=0)
        self.tensorboard.set_model(self.model)

    def _model(self):
        model = Sequential()
        model.add(Dense(units=128, activation="relu", input_dim=self.state_size))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(units=128, activation="relu"))
        model.add(Dense(units=self.output_size))

        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, is_eval=False):
        # take random action in order to diversify experience at the beginning
        if not is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        action_probs = self.model.predict(state)
        return np.argmax(action_probs[0])

    def train_experience_replay(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)
        X_train, y_train = [], []

        # DQN
        for state, action, reward, next_state, done in mini_batch:
            if done:
                target = reward
            else:
                # approximate deep q-learning equation
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            # estimate q-values based on current state
            q_values = self.model.predict(state)
            # update the target for current action based on discounted reward
            q_values[0][action] = target

            X_train.append(state[0])
            y_train.append(q_values[0])

        # update q-function parameters based on huber loss gradient
        loss = self.model.fit(
            np.array(X_train), np.array(y_train),
            epochs=1, verbose=0,
            callbacks=[self.tensorboard] if self.is_training else []
        ).history["loss"][0]

        # as the training goes on we want the agent to
        # make less random and more optimal decisions
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def save(self, episode):
        self.model.save(f"{self.model_path}/{self.model_name}_{episode}")
        self.save_memory(episode)

    def save_memory(self, episode):
        with open(f"{self.model_path}/{self.model_name}_{episode}/memory.json", 'w') as memory_file:
            memory_file.write(json.dumps(
                {
                    # "memory": list(pickle.dumps(self.memory)),
                    "epsilon": self.epsilon
                }
            ))

    def load(self):
        model = load_model(f"{self.model_path}/{self.model_name}", custom_objects=self.custom_objects)
        self.load_memory()
        return model

    def load_memory(self):
        try:
            with open(f"{self.model_path}/{self.model_name}/memory.json", 'r') as memory_file:
                memory_data = json.loads(memory_file.read())
                self.epsilon = memory_data["epsilon"]
        except OSError as e:
            pass
