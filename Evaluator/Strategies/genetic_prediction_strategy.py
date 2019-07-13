"""
OctoBot Tentacle

$tentacle_description: {
    "package_name": "OctoBot-IA-Tentacles",
    "name": "genetic_prediction_evaluator",
    "type": "Evaluator",
    "subtype": "Strategies",
    "version": "1.0.0",
    "requirements": ["genetic_prediction_evaluator, genetic_tools"],
    "config_files": ["GeneticPredictionStrategy.json"],
    "config_schema_files": ["GeneticPredictionStrategy_schema.json"]
}
"""
#  Drakkar-Software OctoBot
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

from config import START_PENDING_EVAL_NOTE, EvaluatorMatrixTypes, TimeFrames, \
    STRATEGIES_REQUIRED_TIME_FRAME
from evaluator.Strategies import MixedStrategiesEvaluator
from tentacles.Evaluator.TA import GeneticPredictionEvaluator


class GeneticPredictionStrategy(MixedStrategiesEvaluator):
    DESCRIPTION = ""
    CANDLES_DATA_CLASS_NAME = GeneticPredictionEvaluator.get_name()

    def __init__(self):
        super().__init__()
        self.time_frame = TimeFrames(self.get_specific_config()[STRATEGIES_REQUIRED_TIME_FRAME][0])

    async def eval_impl(self) -> None:
        try:
            self.eval_note = self.matrix[EvaluatorMatrixTypes.TA][self.CANDLES_DATA_CLASS_NAME][self.time_frame]
        except KeyError as e:
            self.logger.warning(f"Failed to compute evaluation: {e}")
            self.eval_note = START_PENDING_EVAL_NOTE
