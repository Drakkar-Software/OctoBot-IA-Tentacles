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
import numpy as np

from octobot_commons import symbol_util
from octobot_commons.channels_name import OctoBotTradingChannelsName
import octobot_commons.constants as commons_constants
import octobot_evaluators.util as evaluators_util

import octobot_trading.api as trading_api
from octobot_evaluators.evaluators import TAEvaluator
from octobot_tentacles_manager.api.configurator import get_tentacle_config
from tentacles.Evaluator.TA.ddqn_evaluator import DDQNAgent


class DDQNEvaluator(TAEvaluator):
    IS_TRAINING_CONFIG = "is_training"
    ACTIONS = {0: 'Hold', 1: 'Buy', 2: 'Sell'}

    def __init__(self, tentacles_setup_config):
        super().__init__(tentacles_setup_config)
        self.exchange_id = None
        self.exchange_manager = None
        self.ddqn_agent = DDQNAgent(
            is_training=get_tentacle_config(self.tentacles_setup_config, self.__class__)[self.IS_TRAINING_CONFIG])
        self.training_period = 1
        self.returns_across_episodes = []
        self.actions = []
        self.num_experience_replay = 0
        self.last_portfolio_value = None

        self.current_action = None
        self.current_state = None
        self.current_reward = 0

        self.next_state = None

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
        self.current_reward += (profitability_percent - market_profitability_percent) / 100

        done = self.is_done()
        self.ddqn_agent.remember(self.current_state, self.actions, self.current_reward, self.next_state, done)

        # update state
        self.current_state = self.next_state

        # experience replay
        if len(self.ddqn_agent.memory) > self.ddqn_agent.buffer_size:
            self.num_experience_replay += 1
            loss = self.ddqn_agent.experience_replay()
            self.logger.info(
                'Loss: {:.2f}\tAction: {}\treward: {}'.format(loss,
                                                              self.ACTIONS[self.current_action],
                                                              self.current_reward))
            self.ddqn_agent.tensorboard.on_batch_end(self.num_experience_replay,
                                                     {'loss': loss, 'profitability': profitability})

        # save models periodically
        if self.training_period % 5 == 0 or done:
            self.ddqn_agent.save_model(self.training_period)
            self.logger.info('model saved')

        if done:
            self.returns_across_episodes.append(profitability)

        self.training_period += 1

    async def ohlcv_callback(self, exchange: str, exchange_id: str,
                             cryptocurrency: str, symbol: str, time_frame, candle, inc_in_construction_data):
        self.current_reward = 0
        prices = self.get_prices(time_frame, symbol)
        currency, market = symbol_util.split_symbol(symbol)
        self.next_state = self.generate_combined_state(self.training_period,
                                                       self.ddqn_agent.WINDOW_SIZE,
                                                       prices,
                                                       self.get_portfolio_currency(market),
                                                       self.get_portfolio_currency(currency))
        if self.current_state is not None:
            self.actions = self.ddqn_agent.model.predict(self.current_state)[0]
            self.current_action = self.ddqn_agent.act(self.current_state)

            # log evaluation
            self.logger.info(
                'Step: {}\tHold signal: {:.4} \tBuy signal: {:.4} \tSell signal: {:.4}'.format(self.training_period,
                                                                                               self.actions[0],
                                                                                               self.actions[1],
                                                                                               self.actions[2]))
            if self.current_action > 1:
                self.eval_note = self.current_action - 1
            if self.current_action == 1:
                self.eval_note = self.current_action - 2
            if self.current_action == 0:
                self.eval_note = self.current_action
            await self.evaluation_completed(cryptocurrency, symbol, time_frame,
                                            eval_time=evaluators_util.get_eval_time(full_candle=candle,
                                                                                    time_frame=time_frame))

    def is_done(self):
        try:
            if self.exchange_manager is not None and self.exchange_manager.backtesting:
                return self.exchange_manager.backtesting.get_progress() >= 0.999
        except ImportError:
            self.logger.error(f"Can't get current exchange time: requires OctoBot-Trading package installed")
        return False

    def get_prices(self, time_frame, symbol):
        return trading_api.get_symbol_close_candles(self.get_exchange_symbol_data(self.exchange_name,
                                                                                  self.exchange_id,
                                                                                  symbol),
                                                    time_frame, include_in_construction=False)

    def get_portfolio_currency(self, currency):
        return trading_api.get_portfolio_currency(self.exchange_manager,
                                                  currency,
                                                  portfolio_type=commons_constants.PORTFOLIO_AVAILABLE)

    def generate_price_state(self, prices, end_index, window_size):
        start_index = end_index - window_size
        if start_index >= 0:
            period = prices[start_index:end_index + 1]
        else:  # if end_index cannot suffice window_size, pad with prices on start_index
            period = -start_index * [prices[0]] + prices[0:end_index + 1]
        return sigmoid(np.diff(period))

    def generate_portfolio_state(self, price, balance, num_holding):
        return [np.log(price), np.log(balance), np.log(num_holding + 1e-6)]

    def generate_combined_state(self, end_index, window_size, prices, balance, num_holding):
        prince_state = self.generate_price_state(prices, end_index, window_size)
        portfolio_state = self.generate_portfolio_state(prices[end_index], balance, num_holding)
        return np.array([np.concatenate((prince_state, portfolio_state), axis=None)])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))
