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

import math
import numpy as np
import keras.backend as K
import tensorflow as tf


def switch_k_backend_device():
    if K.backend() == "tensorflow":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def huber_loss(y_true, y_pred, clip_delta=1.0):
    """Huber loss - Custom Loss Function for Q Learning
    Links: 	https://en.wikipedia.org/wiki/Huber_loss
            https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
    """
    error = y_true - y_pred
    cond = K.abs(error) <= clip_delta
    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
    return K.mean(tf.where(cond, squared_loss, quadratic_loss))


def sigmoid(v):
    try:
        if v < 0:
            return 1 - 1 / (1 + math.exp(v))
        return 1 / (1 + math.exp(-v))
    except Exception as e:
        print(f"Error when computing sigmoid : {e}")


def get_state(current_price, symbol_holding, market_holding, evaluation_values):
    return np.array([[sigmoid(current_price),
                      sigmoid(symbol_holding),
                      sigmoid(market_holding),
                      *[sigmoid(value) for value in evaluation_values]]],
                    dtype=float)
