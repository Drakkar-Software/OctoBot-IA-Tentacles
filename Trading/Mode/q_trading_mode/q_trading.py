#  Drakkar-Software OctoBot-Tentacles
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
import asyncio
import enum

import numpy as np

import async_channel.constants as channel_constants

import octobot_evaluators.api as evaluators_api
import octobot_evaluators.enums as evaluators_enums
import octobot_evaluators.matrix as matrix
import octobot_trading.constants as trading_constants
import octobot_trading.exchange_channel as exchanges_channel
import octobot_trading.enums as trading_enums
import octobot_trading.modes as trading_modes
from octobot_commons import symbol_util
import octobot_trading.errors as trading_errors
import octobot_trading.personal_data as trading_personal_data
from tentacles.Trading.Mode.q_trading_mode.q_agent import QAgent
from tentacles.Trading.Mode.q_trading_mode.utils import get_state


class ActionOrderType(enum.Enum):
    SIT = 0
    BUY_MARKET = 1
    SELL_MARKET = 2
    BUY_LIMIT = 3
    SELL_LIMIT = 4


class QTradingMode(trading_modes.AbstractTradingMode):

    def __init__(self, config, exchange_manager):
        super().__init__(config, exchange_manager)
        self.load_config()

    def get_current_state(self) -> (str, float):
        return super().get_current_state()[0] if self.producers[0].state is None else self.producers[0].state.name, \
               self.producers[0].final_eval

    async def create_producers(self) -> list:
        mode_producer = QTradingModeProducer(
            exchanges_channel.get_chan(trading_constants.MODE_CHANNEL, self.exchange_manager.id),
            self.config, self, self.exchange_manager)
        await mode_producer.run()
        return [mode_producer]

    async def create_consumers(self) -> list:
        mode_consumer = QTradingModeConsumer(self)
        await exchanges_channel.get_chan(trading_constants.MODE_CHANNEL, self.exchange_manager.id).new_consumer(
            consumer_instance=mode_consumer,
            trading_mode_name=self.get_name(),
            cryptocurrency=self.cryptocurrency if self.cryptocurrency else channel_constants.CHANNEL_WILDCARD,
            symbol=self.symbol if self.symbol else channel_constants.CHANNEL_WILDCARD,
            time_frame=self.time_frame if self.time_frame else channel_constants.CHANNEL_WILDCARD)
        return [mode_consumer]

    @classmethod
    def get_is_symbol_wildcard(cls) -> bool:
        return False


class QTradingModeConsumer(trading_modes.AbstractTradingModeConsumer):
    def __init__(self, trading_mode):
        super().__init__(trading_mode)
        self.trader = self.exchange_manager.trader

    def flush(self):
        super().flush()
        self.trader = None

    async def can_create_order(self, symbol, state):
        return True

    async def create_new_orders(self, symbol, final_note, state, **kwargs):
        order_type, order_price = final_note

        currency, market = symbol_util.split_symbol(symbol)
        current_symbol_holding = self.exchange_manager.exchange_personal_data.portfolio_manager.portfolio \
            .get_currency_portfolio(currency)
        current_market_quantity = self.exchange_manager.exchange_personal_data.portfolio_manager.portfolio \
            .get_currency_portfolio(market)

        self.logger.debug(f"New final note : {order_type} {order_price}")
        if order_type != ActionOrderType.SIT.value:
            try:
                trader_order_type = None
                current_order = None
                if order_type == ActionOrderType.BUY_MARKET.value:
                    trader_order_type = trading_enums.TraderOrderType.BUY_MARKET
                    current_order = trading_personal_data.create_order_instance(trader=self.trader,
                                                                                order_type=trader_order_type,
                                                                                symbol=symbol,
                                                                                current_price=order_price,
                                                                                quantity=current_market_quantity,
                                                                                price=order_price)
                # elif order_type == ActionOrderType.BUY_LIMIT.value:
                #     trader_order_type = trading_enums.TraderOrderType.BUY_LIMIT
                elif order_type == ActionOrderType.SELL_MARKET.value:
                    trader_order_type = trading_enums.TraderOrderType.SELL_MARKET
                    # elif order_type == ActionOrderType.SELL_LIMIT.value:
                    #     trader_order_type = trading_enums.TraderOrderType.SELL_LIMIT
                    current_order = trading_personal_data.create_order_instance(trader=self.trader,
                                                                                order_type=trader_order_type,
                                                                                symbol=symbol,
                                                                                current_price=order_price,
                                                                                quantity=current_symbol_holding,
                                                                                price=order_price)
                await self.trader.create_order(current_order)
                return [current_order]
            except (trading_errors.MissingFunds, trading_errors.MissingMinimalExchangeTradeVolume):
                raise
            except asyncio.TimeoutError as e:
                self.logger.error(
                    f"Impossible to create order for {symbol} on {self.exchange_manager.exchange_name}: {e} "
                    f"and is necessary to compute the order details.")
            except Exception as e:
                self.logger.exception(e, True, f"Failed to create order : {e}.")
        else:
            self.logger.debug("Doesn't create new order : SIT")
        return []


class QTradingModeProducer(trading_modes.AbstractTradingModeProducer):
    MODEL_NAME = "q-trading"
    MODEL_PATH = "models"
    EVALUATION_SIZE = 5
    EVALUATION_COUNT = 1  # RSI
    EVALUATOR_TIME_FRAMES_COUNT = 3  # 5m, 1h, 1d
    FINAL_EVAL_SIZE = EVALUATION_COUNT * EVALUATION_SIZE * EVALUATOR_TIME_FRAMES_COUNT
    WINDOW_SIZE = 1 + FINAL_EVAL_SIZE  # current price + FINAL_EVAL_SIZE
    OUTPUT_SIZE = 3  # SIT, BUY, SELL
    BATCH_SIZE = 32
    EPISODE_TO_SAVE = 50
    IS_TRAINING_CONFIG = "is_training"

    def __init__(self, channel, config, trading_mode, exchange_manager):
        super().__init__(channel, config, trading_mode, exchange_manager)
        trading_config = self.trading_mode.trading_config if self.trading_mode else {}
        self.IS_TRAINING = trading_config.get(self.IS_TRAINING_CONFIG, False)

        self.q_agent = QAgent(
            is_training=self.IS_TRAINING,
            model_name=self.MODEL_NAME,
            model_path=self.MODEL_PATH,
            state_size=self.WINDOW_SIZE,
            output_size=self.OUTPUT_SIZE,
            action_size=3)

        self.total_reward = 0
        self.last_reward = 0
        self.previous_reward = 0
        self.episode = 0

        self.current_state = None
        self.next_state = None

    @classmethod
    def get_should_cancel_loaded_orders(cls):
        return True

    async def stop(self):
        if self.trading_mode is not None:
            self.trading_mode.consumers[0].flush()
        await super().stop()

    async def start(self) -> None:
        await super().start()
        await self.subscribe_balance_profitability_channel()

    async def subscribe_balance_profitability_channel(self):
        await exchanges_channel.get_chan(trading_constants.BALANCE_PROFITABILITY_CHANNEL,
                                         self.exchange_manager.id).new_consumer(
            self.balance_profitability_callback,
            priority_level=self.priority_level)

    async def balance_profitability_callback(self,
                                             exchange,
                                             exchange_id,
                                             profitability,
                                             profitability_percent,
                                             market_profitability_percent,
                                             initial_portfolio_current_profitability,
                                             ):
        self.previous_reward = self.last_reward
        self.last_reward = (profitability_percent - market_profitability_percent) / 100
        self.logger.info(f"Reward updated : last_reward = {self.last_reward} & total_reward = {self.total_reward}")

    async def set_final_eval(self, matrix_id: str, cryptocurrency: str, symbol: str, time_frame):
        for evaluated_strategy_node in matrix.get_tentacles_value_nodes(
                matrix_id,
                matrix.get_tentacle_nodes(matrix_id,
                                          exchange_name=self.exchange_name,
                                          tentacle_type=evaluators_enums.EvaluatorMatrixTypes.STRATEGIES.value),
                cryptocurrency=cryptocurrency,
                symbol=symbol):
            self.final_eval = np.array([])
            self.flatten_eval_notes(evaluators_api.get_value(evaluated_strategy_node))
            if len(self.final_eval) == self.FINAL_EVAL_SIZE:
                mark_price = await self.pre_eval_data(symbol, self.exchange_manager)
                action = self.train(mark_price) if self.IS_TRAINING else self.evaluate(mark_price)
                await self.submit_trading_evaluation(cryptocurrency=cryptocurrency,
                                                     symbol=symbol,
                                                     time_frame=None,
                                                     final_note=[action, mark_price])

    def flatten_eval_notes(self, eval_dict: dict):
        for eval_by_ta in eval_dict.values():
            for evaluation in eval_by_ta.values():
                eval_value = evaluators_api.get_value(evaluation)[-self.EVALUATION_SIZE:]
                if isinstance(eval_value, np.ndarray):
                    self.final_eval = np.concatenate((self.final_eval, eval_value))
                if isinstance(eval_value, int) or isinstance(eval_value, float):
                    self.final_eval = np.append(self.final_eval, eval_value)

    async def pre_eval_data(self, symbol, exchange_manager):
        # currency, market = symbol_util.split_symbol(symbol)
        # current_symbol_holding = self.exchange_manager.exchange_personal_data.portfolio_manager.portfolio \
        #     .get_currency_portfolio(currency)
        # current_market_quantity = self.exchange_manager.exchange_personal_data.portfolio_manager.portfolio \
        #     .get_currency_portfolio(market)

        try:
            mark_price = await exchange_manager.exchange_symbols_data.get_exchange_symbol_data(symbol) \
                .prices_manager.get_mark_price(timeout=trading_constants.ORDER_DATA_FETCHING_TIMEOUT)
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError("Mark price is not available")
        return mark_price

    def train(self, current_price):
        action = ActionOrderType.SIT.value
        self.next_state = get_state(current_price, self.final_eval)

        if self.current_state is not None:
            action = self.q_agent.act(self.current_state)
            reward = self.last_reward - self.previous_reward
            self.total_reward += reward
            self.q_agent.remember(self.current_state, action, reward, self.next_state, self.is_done())
            self.save_loss()
            self.save_if_necessary()

        self.current_state = self.next_state
        return action

    def save_loss(self):
        if len(self.q_agent.memory) > self.BATCH_SIZE:
            loss = self.q_agent.train_experience_replay(self.BATCH_SIZE)
            self.logger.info(f"Trained experience, loss = {loss}")

    def save_if_necessary(self):
        if self.episode % self.EPISODE_TO_SAVE == 0 or self.is_done():
            self.q_agent.save(self.episode)

    def evaluate(self, current_price):
        action = ActionOrderType.SIT.value
        self.next_state = get_state(current_price, self.final_eval)

        if self.current_state is not None:
            action = self.q_agent.act(self.current_state, is_eval=True)
            reward = self.last_reward - self.previous_reward
            self.total_reward += reward
            self.q_agent.memory.append((self.current_state, action, reward, self.next_state, self.is_done()))

        self.current_state = self.next_state
        return action

    def is_done(self):
        if self.exchange_manager is not None and self.exchange_manager.backtesting:
            return self.exchange_manager.backtesting.get_progress() >= 0.999
        return False
