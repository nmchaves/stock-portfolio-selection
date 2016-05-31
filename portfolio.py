import numpy as np

import util
from constants import init_dollars
from market_data import MarketData
from util import empirical_sharpe_ratio
#import matplotlib.pyplot as plt


class Portfolio(object):
    """
    Superclass for any portfolio optimization algorithm.

    Follows the conventions used in most of portfolio optimization
    literature.
    """

    def __init__(self, market_data, market_data_train=None, start=0, stop=None, rebal_interval=1, tune_interval=None, tune_length=None,
                 init_b=None, init_dollars=init_dollars, verbose=False, silent=False,
                 past_results_dir=None, new_results_dir=None, repeat_past=False):
        """
        :param market_data: Stock market data (MarketData object)
        :param start: What day this portfolio starts at
        :param init_b: An initial allocation to make at the end of the 1st day. This is useful if
        you want to inject prior knowledge about which stocks you think will perform well. Also useful
        for tuning hyperparameters, because we may want the portfolio to start out in a particular state.
        :param rebal_interval: Rebalance interval (Rebalance the portfolio every |reb_int| days)
        """

        if not isinstance(market_data, MarketData):
            raise Exception('market_data input to Portfolio constructor must be a MarketData object.')

        self.data = market_data
        self.data_train = market_data_train
        self.num_stocks = len(self.data.stock_names)
        self.start = start

        if stop:
            self.stop = stop
            self.num_days = stop - start
        else:
            last_day = self.data.get_vol().shape[0]
            self.stop = last_day
            self.num_days = last_day - self.start

        self.rebal_interval = rebal_interval  # How often to rebalance
        self.tune_interval = tune_interval  # How often to tune hyperparams (if at all)
        self.b = init_b  # b[i] = Fraction of total money allocated to stock i

        self.b_history = np.zeros((self.num_stocks, self.num_days))  # portfolio before open of each day
        self.dollars_op_history = np.zeros(self.num_days)
        self.dollars_op_history[0] = init_dollars
        self.dollars_cl_history = np.zeros(self.num_days)  # Dollars before close each day
        self.last_close_price = np.NaN * np.ones(self.num_stocks)
        self.sharpe = None  # Sharpe ratio. Calculate after finished running
        self.verbose = verbose
        self.silent = silent

        self.new_results_dir = new_results_dir

        if past_results_dir is not None:
            past_b_history, past_dollars_history = self.load_previous_results(past_results_dir)
            self.past_b_history = past_b_history
            self.past_dollars_history = past_dollars_history
            self.len_past = past_b_history.shape[1]
            self.b = past_b_history[:, -1]  # Use previous b as initialization (overrides |init_b| argument)
        else:
            self.past_b_history = None
            self.past_dollars_history = None

        self.repeat_past = repeat_past

    def tune_hyperparams(self, cur_day):
        # Implement this in your portfolio if you want to tune
        raise 'tune_hyperparams is an abstract method, so it must be implemented by the child class!'

    def update(self, cur_day, init=False):
        """
        Update the portfolio

        :param cur_day: 0-based index of today's date
        :param init: If True, this portfolio is being initialized today.
        :return: None
        """

        # Check if we need to tune hyperparameters today
        if self.tune_interval and not self.repeat_past:
            if cur_day > self.start and cur_day % self.tune_interval == 0:
                self.tune_hyperparams(cur_day)

        self.update_allocation(cur_day, init)
        self.update_dollars(cur_day)

        return

    def update_allocation(self, cur_day, init=False):
        """

        :param cur_day:
        :param init: If True, this portfolio is being initialized today.
        :return:
        """

        day_idx = cur_day - self.start

        if self.repeat_past:
            # Use results we've already run w/out re-running algorithm
            if day_idx < self.len_past-1:
                self.b = self.past_b_history[:, day_idx+1]
            return

        if init and (self.b is not None):
            # b has already been initialized using initialization argument init_b
            # This may be useful for the test set where we may not want to initialize uniformly.
            self.b_history[:, day_idx+1] = self.b
            return

        if (cur_day % self.rebal_interval) != 0:
            # Don't make any trades today (avoid transaction costs)
            # TODO: need to use special flags to indicate hold when using Yanjun's framework.
            return

        self.b = self.get_new_allocation(cur_day, init)
        return

    def get_new_allocation(self, cur_day, init=False):
        raise 'get_new_allocation is an abstract method, so it must be implemented by the child class!'


    def update_dollars(self, cur_day):
        """

        :param op:
        :param cl:
        :param cur_dollars: Dollars before trading (at the open prices)
        :return:
        """

        day_idx = cur_day - self.start  # DON'T use this for accessing market data (use absolute date for market data)

        prev_cl = self.last_close_price
        op = self.data.get_op(relative=False)[cur_day, :]
        cl = self.data.get_cl(relative=False)[cur_day, :]
        new_portfolio = self.b

        # Get the value of our portfolio at the end of Day t before paying transaction costs
        isActive = np.isfinite(op)
        value_vec = self.dollars_op_history[day_idx] * self.b_history[:, day_idx]
        growth = cl[isActive] / prev_cl[isActive]-1
        growth[np.isnan(growth)] = 0
        revenue_vec = value_vec[isActive] * growth
        value_vec[isActive] = value_vec[isActive] + revenue_vec
        self.dollars_cl_history[day_idx] = self.dollars_op_history[day_idx] + np.sum(revenue_vec)

        # At the end of Day t, we use the close price of day t to adjust our
        # portfolio to the desired percentage.
        if day_idx <= self.num_days-2:
            nonActive = np.logical_not(isActive)
            value_realizable = self.dollars_cl_history[day_idx] - np.sum(value_vec[nonActive])
            new_value_vec, trans_cost = util.rebalance(value_vec[isActive], value_realizable,
                                                       new_portfolio[isActive])

            self.dollars_op_history[day_idx+1] = self.dollars_cl_history[day_idx] - trans_cost
            value_vec[isActive] = new_value_vec
            self.b_history[:, day_idx+1] = value_vec / self.dollars_op_history[day_idx+1]

        self.last_close_price[isActive] = cl[isActive]
        return

    def run(self, start=None, stop=None):
        """

        :param start:
        :param stop:
        :return: None
        """

        if start is None:
            start = self.start
        if stop is None:
            stop = self.stop

        for day in range(start, stop):
            if day == start:
                init = True
            else:
                init = False
            self.update(day, init)
        self.sharpe = empirical_sharpe_ratio(self.dollars_op_history)

        self.print_results()
        if self.new_results_dir is not None:
            self.save_results()


    def print_results(self):
        if self.verbose:
            print 'Total dollar value of assets over time:'
            print self.dollars_op_history[0:30]
            #plt.plot(self.dollars_op_history)
            #plt.show()

        if not self.silent:
            print 'Sharpe ratio:'
            print self.sharpe

    def save_results(self):
        if self.new_results_dir is None:
            return

        print 'Saving ', self.portfolio_type
        save_dir = self.new_results_dir

        util.save_dollars_history(save_dir=save_dir, dollars=self.dollars_op_history, portfolio_type=self.portfolio_type)
        util.save_b_history(save_dir=save_dir, b_history=self.b_history, portfolio_type=self.portfolio_type)
        util.save_hyperparams(save_dir=save_dir, hyperparams_dict=self.get_hyperparams_dict(), portfolio_type=self.portfolio_type)
        return

    def get_hyperparams_dict(self):
        raise 'Abstract method. Implement in the child class.'

    def load_previous_results(self, past_results_dir):

        # Load past dollars history
        past_dollars_op_history = np.loadtxt(past_results_dir + 'dollars_history.txt', delimiter='\t')

        # Load past portfolio history
        past_b_history = np.loadtxt(past_results_dir + 'b_history.txt', delimiter='\t')

        return past_b_history, past_dollars_op_history

    """
    Getters
    """
    def get_b(self):
        return self.b

    def get_dollars_history(self):
        return self.dollars_op_history
