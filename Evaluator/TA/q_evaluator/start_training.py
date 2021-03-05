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
from tqdm import tqdm

from octobot.cli import start_background_octobot_with_args

BACKTESTING_FILES = ["ExchangeHistoryDataCollector_1589016793.8005197.data",
                     "ExchangeHistoryDataCollector_1588405922.6261444.data",
                     "ExchangeHistoryDataCollector_1590165463.2795594.data",
                     "ExchangeHistoryDataCollector_1610816291.8282602.data",
                     "ExchangeHistoryDataCollector_1614855165.3938067.data",
                     "ExchangeHistoryDataCollector_1586593427.2594533.data",
                     "ExchangeHistoryDataCollector_1586598990.0092158.data"]

COMPLETE_RUN = 2

if __name__ == '__main__':
    try:
        for i in range(COMPLETE_RUN):
            for backtesting_file in tqdm(BACKTESTING_FILES):
                bot_process = start_background_octobot_with_args(
                    backtesting=True,
                    simulate=False,
                    backtesting_files=[backtesting_file],
                    in_subprocess=True
                )
                bot_process.join()
    except KeyboardInterrupt:
        pass
