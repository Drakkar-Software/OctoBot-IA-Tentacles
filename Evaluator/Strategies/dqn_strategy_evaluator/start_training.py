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
from argparse import Namespace

from tqdm import tqdm

from start import start_octobot

BACKTESTING_FILES = ["ExchangeHistoryDataCollector_1587760604.1586962.data",
                     "ExchangeHistoryDataCollector_1588624733.7239532.data",
                     "ExchangeHistoryDataCollector_1589300711.251622.data",
                     "ExchangeHistoryDataCollector_1589300724.2862911.data",
                     "ExchangeHistoryDataCollector_1589300734.8648586.data",
                     "ExchangeHistoryDataCollector_1589300745.8589752.data"]
COMPLETE_RUN = 2

if __name__ == '__main__':
    for i in range(COMPLETE_RUN):
        for backtesting_file in tqdm(BACKTESTING_FILES):
            args = Namespace(version=False,
                             encrypter=False,
                             strategy_optimizer=False,
                             data_collector=False,
                             backtesting_files=[backtesting_file],
                             no_telegram=False,
                             no_web=False,
                             backtesting=True,
                             simulate=False,
                             risk=None)
            start_octobot(args)
