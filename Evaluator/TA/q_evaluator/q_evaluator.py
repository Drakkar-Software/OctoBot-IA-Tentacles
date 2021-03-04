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
import enum
import numpy as np
import tqdm

import octobot_commons.constants as commons_constants
import octobot_evaluators.util as evaluators_util
import octobot_trading.api as trading_api
from octobot_commons import symbol_util
from octobot_commons.channels_name import OctoBotTradingChannelsName
from octobot_evaluators.evaluators import TAEvaluator
from octobot_tentacles_manager.api.configurator import get_tentacle_config
import tentacles.Evaluator.TA.q_evaluator as q_evaluator


class Actions(enum.Enum):
    SIT = 0
    BUY = 1
    SELL = 2


class QEvaluator(TAEvaluator):
    MODEL_NAME = "q-evaluator"
    MODEL_PATH = "models"
    WINDOW_SIZE = 10
    BATCH_SIZE = 32
    EPISODE_COUNT = 100
    IS_TRAINING_CONFIG = "is_training"

    def __init__(self, tentacles_setup_config):
        super().__init__(tentacles_setup_config)
        self.exchange_id = None
        self.exchange_manager = None
        self.q_agent = q_evaluator.QAgent(
            is_training=get_tentacle_config(self.tentacles_setup_config, self.__class__)[self.IS_TRAINING_CONFIG],
            model_name=self.MODEL_NAME,
            model_path=self.MODEL_PATH,
            state_size=self.WINDOW_SIZE)

        self.total_reward = 0
        self.last_reward = 0
        self.previous_reward = 0
        self.episode = 0

    async def start(self, bot_id: str) -> bool:
        await super().start(bot_id)
        try:
            from octobot_trading.exchange_channel import get_chan as get_trading_chan
            from octobot_trading.api.exchange import get_exchange_id_from_matrix_id, \
                get_exchange_manager_from_exchange_id
            self.exchange_id = get_exchange_id_from_matrix_id(self.exchange_name, self.matrix_id)
            self.exchange_manager = get_exchange_manager_from_exchange_id(self.exchange_id)
            await get_trading_chan(OctoBotTradingChannelsName.BALANCE_PROFITABILITY_CHANNEL.value, self.exchange_id) \
                .new_consumer(self.balance_profitability_callback, priority_level=self.priority_level)
            return True
        except ImportError as e:
            self.logger.error("Can't connect to Profitability trading channel")
        return False

    async def balance_profitability_callback(self,
                                             exchange: str,
                                             exchange_id: str,
                                             profitability,
                                             profitability_percent,
                                             market_profitability_percent,
                                             initial_portfolio_current_profitability,
                                             ):
        self.previous_reward = self.last_reward
        self.last_reward = (profitability_percent - market_profitability_percent) / 100
        self.logger.info(f"Reward updated : last_reward = {self.last_reward} & total_reward = {self.total_reward}")

    async def ohlcv_callback(self, exchange: str, exchange_id: str,
                             cryptocurrency: str, symbol: str, time_frame, candle, inc_in_construction_data):
        self.episode += 1
        prices = self.get_prices(time_frame, symbol)
        self.train(prices)
        action, history = self.evaluate(prices)

        if action == Actions.BUY.value:
            self.eval_note = -1
        if action == Actions.SELL.value:
            self.eval_note = 1
        if action == Actions.SIT.value:
            self.eval_note = 0
        await self.evaluation_completed(cryptocurrency, symbol, time_frame,
                                        eval_time=evaluators_util.get_eval_time(full_candle=candle,
                                                                                time_frame=time_frame))

    def get_prices(self, time_frame, symbol):
        return trading_api.get_symbol_close_candles(self.get_exchange_symbol_data(self.exchange_name,
                                                                                  self.exchange_id,
                                                                                  symbol),
                                                    time_frame, include_in_construction=False)

    def train(self, data):
        data_length = len(data) - 1
        self.q_agent.inventory = []
        avg_loss = []
        state = q_evaluator.get_state(data, 0, self.WINDOW_SIZE + 1)

        for t in tqdm.tqdm(range(data_length), total=data_length, leave=True,
                           desc=f"Episode {self.episode}/{self.EPISODE_COUNT}"):
            reward = 0
            next_state = q_evaluator.get_state(data, t + 1, self.WINDOW_SIZE + 1)

            # select an action
            action = self.q_agent.act(state)

            if action == Actions.BUY.value:
                self.q_agent.inventory.append(data[t])
            elif action == Actions.SELL.value and len(self.q_agent.inventory) > 0:
                reward = self.last_reward - self.previous_reward
                self.total_reward += reward
                self.q_agent.inventory.pop(0)
            else:
                pass

            done = (t == data_length - 1)
            self.q_agent.remember(state, action, reward, next_state, done)

            if len(self.q_agent.memory) > self.BATCH_SIZE:
                loss = self.q_agent.train_experience_replay(self.BATCH_SIZE)
                avg_loss.append(loss)

            state = next_state

        if self.episode % 10 == 0:
            self.q_agent.save(self.episode)

        return np.mean(np.array(avg_loss))

    def evaluate(self, data):
        data_length = len(data) - 1
        history = []
        self.q_agent.inventory = []
        state = q_evaluator.get_state(data, 0, self.WINDOW_SIZE + 1)

        for t in range(data_length):
            reward = 0
            next_state = q_evaluator.get_state(data, t + 1, self.WINDOW_SIZE + 1)
            action = self.q_agent.act(state, is_eval=True)

            if action == Actions.BUY.value:
                self.q_agent.inventory.append(data[t])
                history.append((data[t], "BUY"))

            elif action == Actions.SELL.value and len(self.q_agent.inventory) > 0:
                self.q_agent.inventory.pop(0)
                reward = self.last_reward - self.previous_reward
                self.total_reward += reward
                history.append((data[t], "SELL"))
            else:
                history.append((data[t], "HOLD"))

            done = (t == data_length - 1)
            self.q_agent.memory.append((state, action, reward, next_state, done))

            state = next_state
            if done:
                return action, history
