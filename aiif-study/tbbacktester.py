import numpy as np
import pandas as pd

import backtesting as bt


class TBBacktester(bt.BacktestingBase):

    def _reshape(self, state):
        return np.reshape(state, [1, self.env.lags, self.env.n_features])

    def backtest_strategy(self):
        self.units = 0
        self.position = 0
        self.trades = 0
        self.current_balance = self.init_amount
        self.net_wealths = list()

        for bar in range(self.env.lags, len(self.env.data)):
            date, price = self.get_date_price(bar)
            if self.trades == 0:
                print(50 * '=')
                print(f'{date} | *** START BACKTEST ***')
                self.print_balance(bar)
                print(50 * '=')
            state = self.env.get_state(bar)
            action = np.argmax(self.model.predict(
                self._reshape(state.values), verbose=0)[0, 0]),
            position = 1 if action == 1 else -1

            if self.position in [0, -1] and position == 1:
                if self.verbose:
                    print(50 * '-')
                    print(f'{date} | *** GOIN LONG ***')
                if self.position == -1:
                    self.place_buy_order(bar - 1, units=-self.units)
                self.place_buy_order(bar - 1, amount=self.current_balance)
                if self.verbose:
                    self.print_net_wealth(bar)
                self.position = 1
            elif self.position in [0, 1] and position == -1:
                if self.verbose:
                    print(50 * '-')
                    print(f'{date} | *** GOING SHORT ***')
                if self.position == 1:
                    self.place_sell_order(bar - 1, units=self.units)
                self.place_sell_order(bar - 1, amount=self.current_balance)
                if self.verbose:
                    self.print_net_wealth(bar)
                self.position = -1
            self.net_wealths.append((date, self.calculate_net_wealth(price)))

        self.net_wealths = pd.DataFrame(self.net_wealths, columns=['date', 'net_wealth'])
        self.net_wealths.set_index('date', inplace=True)
        self.net_wealths.index = pd.DatetimeIndex(self.net_wealths.index)
        self.close_out(bar)
