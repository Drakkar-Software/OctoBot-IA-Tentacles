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
import time
from copy import deepcopy

import numpy as np
from octobot_commons.channels_name import OctoBotTradingChannelsName
from octobot_commons.constants import START_PENDING_EVAL_NOTE
from octobot_commons.evaluators_util import check_valid_eval_note
from octobot_evaluators.api.matrix import get_value, get_type
from octobot_evaluators.constants import EVALUATOR_EVAL_DEFAULT_TYPE
from octobot_evaluators.data_manager.matrix_manager import get_evaluations_by_evaluator
from octobot_evaluators.enums import EvaluatorMatrixTypes
from octobot_evaluators.evaluator import StrategyEvaluator
from octobot_tentacles_manager.api.configurator import get_tentacle_config
from octobot_tentacles_manager.loaders.tentacle_loading import get_resources_path

from tentacles.Evaluator.Util.keras import KerasAgent


class DQNStrategyEvaluator(StrategyEvaluator):
    MODEL_NAME = "test"
    IS_TRAINING_CONFIG = "is_training"
    WINDOW_SIZE = 1

    def __init__(self):
        super().__init__()
        self.model_path = os.path.join(get_resources_path(self.__class__), self.MODEL_NAME)
        self.is_training = get_tentacle_config(self.__class__)[self.IS_TRAINING_CONFIG]
        self.window_size = self.WINDOW_SIZE

        self.agent = KerasAgent(self.window_size, model_path=self.model_path, is_training=self.is_training)
        self.batch_size = 32

        self.state = None
        self.next_state = None
        self.action = 0
        self.agent.inventory = []
        self.last_profitability = 0

        # learning attributes
        self.action_history = []
        self.start_time = time.time()

    async def start(self, bot_id: str) -> bool:
        await super().start(bot_id)
        try:
            from octobot_trading.channels.exchange_channel import get_chan as get_trading_chan
            from octobot_trading.api.exchange import get_exchange_id_from_matrix_id
            exchange_id = get_exchange_id_from_matrix_id(self.exchange_name, self.matrix_id)
            await get_trading_chan(OctoBotTradingChannelsName.BALANCE_PROFITABILITY_CHANNEL.value, exchange_id) \
                .new_consumer(self.balance_profitability_callback, priority_level=self.priority_level)
            return True
        except ImportError:
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
        done = False
        try:
            from octobot_trading.api.exchange import get_exchange_manager_from_exchange_name_and_id
            if self.is_training and exchange_id is not None:
                done = get_exchange_manager_from_exchange_name_and_id(exchange,
                                                                      exchange_id).backtesting.get_progress() >= 0.999
        except ImportError:
            self.logger.error(f"Can't get current exchange time: requires OctoBot-Trading package installed")

        if self.next_state is None:
            return

        if self.state is None:
            self.state = self.next_state

        reward = self.last_profitability + profitability
        self.agent.memory.append((self.state, self.action, reward, self.next_state, done))
        self.last_profitability = profitability
        self.state = deepcopy(self.next_state)

        if len(self.agent.memory) > self.batch_size:
            self.agent.exp_repay(self.batch_size)

        if done:
            print(self.action_history)
            print(profitability_percent)
            print(f"Done in {time.time() - self.start_time}s")
            if self.is_training:
                print("Done, saving model...")
                self.agent.model.save(self.model_path)

    async def matrix_callback(self,
                              matrix_id,
                              evaluator_name,
                              evaluator_type,
                              eval_note,
                              eval_note_type,
                              exchange_name,
                              cryptocurrency,
                              symbol,
                              time_frame):
        if evaluator_type == EvaluatorMatrixTypes.TA.value:
            self.eval_note = START_PENDING_EVAL_NOTE
            TA_by_timeframe = {
                available_time_frame: get_evaluations_by_evaluator(
                    matrix_id,
                    exchange_name,
                    EvaluatorMatrixTypes.TA.value,
                    cryptocurrency,
                    symbol,
                    available_time_frame.value,
                    allow_missing=False,
                    allowed_values=[START_PENDING_EVAL_NOTE])
                for available_time_frame in self.strategy_time_frames
            }

            try:
                input_data = []
                for time_frame, eval_by_ta in TA_by_timeframe.items():
                    for evaluation in eval_by_ta.values():
                        eval_value = get_value(evaluation)
                        if check_valid_eval_note(eval_value, eval_type=get_type(evaluation),
                                                 expected_eval_type=EVALUATOR_EVAL_DEFAULT_TYPE):
                            input_data.append(eval_note)
                        else:
                            input_data.append(0)

                self.next_state = np.array(input_data)
                if self.state is not None:
                    self.action = self.agent.act(self.state)
                    self.action_history.append(self.action)
                    if self.action == 1:
                        self.eval_note = -1
                    if self.action == 2:
                        self.eval_note = 1
                    await self.strategy_completed(cryptocurrency, symbol)
            except KeyError as e:
                self.logger.error(f"Missing required evaluator: {e}")
