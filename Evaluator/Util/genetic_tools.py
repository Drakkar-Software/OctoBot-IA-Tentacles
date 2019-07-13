"""
OctoBot Tentacle

$tentacle_description: {
    "package_name": "OctoBot-IA-Tentacles",
    "package_name": "OctoBot-IA-Tentacles",
    "name": "genetic_tools",
    "type": "Evaluator",
    "subtype": "Util",
    "version": "1.0.0",
    "requirements": []
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
import operator
import random
from functools import reduce

from random import randint as ri
from random import uniform as ru

from octobot_commons.enums import PriceIndexes

random.seed(1234)


def percent_diff(approx, correct):
    # print "approx, correct"
    # print approx, correct
    if correct > 0:
        return (approx - correct) / float(correct)
    else:
        return 0


class Individual:
    def __init__(self, number_of_signals, bit_length=16):
        self.rep = ""
        self.sig_size = bit_length
        # self.signal_range = signal_ranges(datasets)
        for i in range(number_of_signals):
            random_val = bin(ri(0, 2 ** bit_length))[2:]
            random_val = random_val.zfill(bit_length)
            self.rep = self.rep + random_val

    def change_individual(self, bit_string):
        if bit_string != None and type(bit_string) is str:
            self.rep = bit_string
        else:
            print("incorrect input")

    def transform_individual(self, signal_ranges):
        """ Takes a list of 2 tuples defining the signal ranges and scales the binary string to the appropriate value
        """
        num_signals = len(signal_ranges)
        repr_size = len(self.rep)
        n = int(repr_size / num_signals)

        individual_signals = [self.rep[i:i + n] for i in range(0, len(self.rep), n)]
        individual_vals = []

        for i in range(num_signals):
            min_val = signal_ranges[i][0]
            max_val = signal_ranges[i][1]
            range_val = abs(max_val - min_val)
            signal_value = min_val + (range_val * (float(int(individual_signals[i], 2)) / 2 ** n))
            individual_vals.append(signal_value)

        return individual_vals

    def mutate(self, mutate_probability):
        l = list(self.rep)

        for i in range(self.sig_size):
            r = ru(0, 1)
            if r < mutate_probability:
                for j in range(i, self.sig_size, self.sig_size):
                    if l[j] == '1':
                        l[j] = '0'
                    elif l[j] == '0':
                        l[j] = '1'

        self.rep = "".join(l)

    def crossover_2way(self, other_parent, crossover_probability):
        l1 = len(self.rep)
        l2 = len(other_parent.rep)
        expected_length = 0
        # print "before crossover"
        # print self.rep
        # print other_parent.rep
        if l1 == l2:
            expected_length = l1
            tmp_p1 = ""
            tmp_p2 = ""
            for i in range(0, l1, self.sig_size):
                p1 = self.rep[i: (i + self.sig_size)]
                p2 = other_parent.rep[i: (i + self.sig_size)]
                point = ri(0, self.sig_size - 1)
                # print point
                tmp1 = p1[point:]
                tmp2 = p2[point:]
                p1 = p1[:point] + tmp2
                p2 = p2[:point] + tmp1
                tmp_p1 = tmp_p1 + p1
                tmp_p2 = tmp_p2 + p2
            # print "After crossover"
            # print tmp_p1
            # print tmp_p2
            # used parent replaces children survivor selection
            if expected_length != 0:
                self.change_individual(tmp_p1.zfill(expected_length))
                other_parent.change_individual(tmp_p2.zfill(expected_length))
            # print "After crossover"
            # print self.rep
            # print other_parent.rep
        else:
            print("length of both parents must be equal")


class Population(object):
    def __init__(self, size, signal_ranges, data, pm=1 / 16.0, pc=0.7):
        individual_size = len(signal_ranges)
        self.population = []
        self.signal_ranges = signal_ranges
        self.pm = pm
        self.pc = pc
        self.size = size
        self.num_signals = len(signal_ranges)
        self.data = data
        # print "self.data"
        # print self.data
        self.data_size = len(data)
        for i in range(size):
            self.population.append(Individual(individual_size))

    def __getitem__(self, key):
        return self.population[key]

    def fitness_function(self, individual):
        transformed_individual = individual.transform_individual(self.signal_ranges)
        # print "Transformed individual"
        # print transformed_individual
        positive = 0
        count = 0
        for element in self.data:
            # one_iter = []
            # print "element"
            # print element
            for item in range(self.num_signals):
                # for item in xrange(3):

                # threshold = int(0.6 * self.num_signals)
                # print "GA, value"
                # print transformed_individual[item]
                # print "actual value"
                # print element[item]
                # print "Delta"
                # print transformed_individual[item] - element[item]
                # one_iter.append(percent_diff(transformed_individual[item], element[item]))
                # print transformed_individual[item]
                # print "element[item]"
                # print element[item]
                # print "abs percent diff"
                # print percent_diff(transformed_individual[item], element[item])
                if abs(percent_diff(transformed_individual[item], element[item])) < 0.5:
                    count = count + 1
                    # print positive
        # return float(positive) / self.data_size
        # return float(positive)
        # print one_iter
        # print "count"
        # print count
        total_items = self.num_signals * self.data_size
        val = float(count) / total_items
        return val

    def sus_sampler(self, fitness_list):
        size = len(fitness_list)
        total_fitness = reduce(operator.add, fitness_list)

        if total_fitness == 0.0:
            print("total_fitness == 0")
            sel_prob_list = fitness_list
        else:
            sel_prob_list = [x / total_fitness for x in fitness_list]
        summed_prob_list = []
        for i in range(1, len(sel_prob_list) + 1):
            summed_prob_list.append(reduce(operator.add, sel_prob_list[:i]))
        # print summed_prob_list

        i = 0
        r = random.uniform(0, 1.0 / size)
        mating_pool = []
        while len(mating_pool) <= size - 1:
            while r < summed_prob_list[i]:
                mating_pool.append(self.population[i])
                r = r + 1.0 / size
            i += 1
        self.population = mating_pool

    def mutation(self):
        for individual in self.population:
            individual.mutate(self.pm)

    def crossover(self):
        for i in range(0, self.size, 2):
            self.population[i].crossover_2way(self.population[i + 1], self.pc)


class FitnessDataObject:
    """ Data used to evaluate the fitness of the evolved model """

    def __init__(self):
        self.data = {}

    def add_signal(self, name, value=0.0):
        if type(name) is str and name not in self.data:
            self.data[name] = value
        else:
            print("Signal name isn't a string")

    def delete_signal(self, name):
        if type(name) is str and name in self.data:
            del self.data[name]
        else:
            print("Signal name isn't a string or signal doesn't exist")

    def get_signals(self):
        return self.data.keys()

    def add_signals(self, names):
        if names:
            for name in names:
                self.add_signal(name)

    def put_value(self, signal_name, value):
        self.data[signal_name] = value

    def get(self, signal_name):
        return self.data[signal_name]


class GeneticDataHandler:

    @staticmethod
    def percent_difference(value, orig):
        return (value - orig) / float(orig)

    def get_dataset(self):
        for i in range(1, len(self.candle_data[PriceIndexes.IND_PRICE_CLOSE.value])):
            model = FitnessDataObject()
            open_diff = self.percent_difference(self.candle_data[PriceIndexes.IND_PRICE_OPEN.value][i],
                                                self.candle_data[PriceIndexes.IND_PRICE_OPEN.value][i - 1])
            high_diff = self.percent_difference(self.candle_data[PriceIndexes.IND_PRICE_HIGH.value][i],
                                                self.candle_data[PriceIndexes.IND_PRICE_HIGH.value][i - 1])
            low_diff = self.percent_difference(self.candle_data[PriceIndexes.IND_PRICE_LOW.value][i],
                                               self.candle_data[PriceIndexes.IND_PRICE_LOW.value][i - 1])
            close_diff = self.percent_difference(self.candle_data[PriceIndexes.IND_PRICE_CLOSE.value][i],
                                                 self.candle_data[PriceIndexes.IND_PRICE_CLOSE.value][i - 1])
            volume_diff = self.percent_difference(self.candle_data[PriceIndexes.IND_PRICE_VOL.value][i],
                                                  self.candle_data[PriceIndexes.IND_PRICE_VOL.value][i - 1])

            # print nasdaq_stock_prices[i].volume_
            # print nasdaq_stock_prices[i-1].volume_
            # print  (nasdaq_stock_prices[i].volume_ -  nasdaq_stock_prices[i-1].volume_) / float(nasdaq_stock_prices[i -1].volume_)

            # volume_diff = 100 *(nasdaq_stock_prices[i].volume_ - nasdaq_stock_prices[i-1].volume_) /  nasdaq_stock_prices[i - 1].volume_
            # print volume_diff
            # dates_ = nasdaq_stock_prices[i].date_

            model.add_signal("delta_open", open_diff)
            model.add_signal("delta_high", high_diff)
            model.add_signal("delta_low", low_diff)
            model.add_signal("delta_close", close_diff)
            model.add_signal("delta_volume", volume_diff)
            self.dataset.append(model)

    def __init__(self, candle_data):
        self.candle_data = candle_data
        self.dataset = []
        self.get_dataset()

    def signal_ranges(self):
        signal_ranges = []
        if self.dataset:
            for signal in self.dataset[0].get_signals():
                l = [dataset.get(signal) for dataset in self.dataset]
                min_max = (min(l), max(l))
                signal_ranges.append(min_max)
        else:
            print("Dataset is None")
        return signal_ranges

    def get_signal_list(self):
        signal_data = []
        for i in range(len(self.dataset)):
            vals = []
            for signal in self.dataset[i].get_signals():
                val = self.dataset[i].get(signal)
                vals.append(val)
            signal_data.append(vals)
        return signal_data
