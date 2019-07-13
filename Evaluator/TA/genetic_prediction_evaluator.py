"""
OctoBot Tentacle

$tentacle_description: {
    "package_name": "OctoBot-IA-Tentacles",
    "name": "genetic_prediction_evaluator",
    "type": "Evaluator",
    "subtype": "TA",
    "version": "1.0.0",
    "requirements": ["genetic_tools"]
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
from evaluator.TA.TA_evaluator import TrendEvaluator
import numpy as np

from tentacles.Evaluator.Util import Population, GeneticDataHandler


class GeneticPredictionEvaluator(TrendEvaluator):
    DESCRIPTION = "Returns the open and close prices of previous candles."
    PAST_CANDLES_COUNT = 5

    async def eval_impl(self):
        genetic_data_handler = GeneticDataHandler(self.data)
        # dataHandler = DataHandler(['apple_data.csv', 'NASDAQ_data.csv'])
        data = genetic_data_handler.get_signal_list()
        signal_ranges = genetic_data_handler.signal_ranges()

        data_element = len(data[0])
        # print "data[0]"
        # print data[0]

        population = Population(100, signal_ranges, data)
        # for i in population:
        #    print i.transform_individual(signal_ranges)
        initial_fitess_values = map(population.fitness_function, population)
        # print initial_fitess_values

        number_generations = 100

        population.sus_sampler(initial_fitess_values)
        max_list = []
        mean_list = []

        for i in range(number_generations):
            # print "initial population"
            # for i in population:
            #    print i.rep
            # print "before crossover"
            population.crossover()
            # print "after crossover"
            # for i in population:
            #    print i.rep
            # print "before mutation"
            population.mutation()
            # for i in population:
            #    print i.rep
            # print "after mutation"
            fitness_vals = map(population.fitness_function, population)
            population.sus_sampler(fitness_vals)
            mean_val = np.mean(fitness_vals)
            max_val = max(fitness_vals)
            print(str(mean_val) + "," + str(max_val))
            mean_list.append(mean_val)
            max_list.append(max_val)
