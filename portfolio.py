"""
    Superclass for any portfolio optimization algorithm.

"""

import numpy as np
from util import get_uniform_allocation, empirical_sharpe_ratio
from constants import init_dollars, cost_per_dollar
from market_data import MarketData


class Portfolio(object):
    def __init__(self, market_data):
        if not isinstance(market_data, MarketData):
            raise 'market_data input to Portfolio constructor must be a MarketData object.'

        self.data = market_data
        self.num_stocks = len(self.data.stock_names)
        self.num_days = self.data.get_vol().shape[0]

        self.b = None # self.init_portfolio_uniform()  # b[i] = Percent of total money allocated to stock i
        self.b_history = []  # History of allocations over time
        self.dollars = init_dollars
        self.dollars_history = []

    def get_b(self):
        return self.b

    def run(self):
        raise 'run is an abstract method, so it must be implemented by the child class!'

    def update(self, cur_day):
        """
        Update the portfolio.

        :param cur_day: 0-based index of today's date
        :return: None
        """
        self.update_allocation(cur_day)

        prev_dollars = self.dollars
        self.dollars_history.append(prev_dollars)
        self.dollars = self.get_dollars(cur_day)

    def update_allocation(self, cur_day):
        if cur_day != 0:
            self.b_history.append(self.b)

        self.b = self.get_new_allocation(cur_day)

    def get_new_allocation(self, cur_day):
        raise 'get_new_allocation is an abstract method, so it must be implemented by the child class!'

    def init_portfolio_uniform(self):
        """
        This function initializes the allocations by naively investing equal
        amounts of money into each stock.
        """
        day_1_op = self.data.get_op(relative=False)[0, :]  # raw opening prices on 1st day
        return get_uniform_allocation(self.num_stocks, day_1_op)

    def get_dollars(self, cur_day, prev_b=None, cur_b=None, cpr=None):
        """
        Calculate the portfolio's wealth for the end of |cur_day| after buying/selling stocks
        at their closing prices.

        :param cur_day: Current day (0-based s.t. cur_day=0 corresponds to the 1st day)
        :param prev_b: Allocation at the end of |cur_day|-1. If not specified, then obtain the
        allocation from self.b_history
        :param cur_b: Allocation at the end of |cur_day|. If not specified, then obtain the
        allocation from self.b
        :param cpr: Closing price relatives for the end of |cur_day|. If not specified, then
        obtain the price relatives using self.data
        :return: The new # of dollars held
        """

        if cur_day == 0:
            # Only buy stocks on day 0 (no selling)
            trans_costs = self.dollars * cost_per_dollar
            return self.dollars - trans_costs

        if cpr is None:
            cpr = np.nan_to_num(self.data.get_cl(relative=True)[cur_day, :])  # closing price relatives
        if cur_b is None:
            cur_b = self.b
        if prev_b is None:
            prev_b = self.b_history[cur_day-1]

        prev_dollars = self.dollars  # amount of money after making trades at end of prev day
        dollars_before_trading = prev_dollars * np.dot(prev_b, cpr)
        if dollars_before_trading <= 0:
            print 'The UCR portfolio ran out of money on day ', str(cur_day), '!'
            exit(0)

        L1_dist = np.linalg.norm((prev_b - cur_b), ord=1)  # L1 distance between new and old allocations
        dollars_trading = dollars_before_trading * L1_dist  # # of dollars that need to be traded to
                                        # reallocate (this already includes costs of both buying and selling!)
        new_dollars = dollars_before_trading - dollars_trading * cost_per_dollar

        if new_dollars <= 0:
            print 'The UCR portfolio ran out of money on day ', str(cur_day), '!'
            exit(0)
        else:
            return new_dollars

    def predict_performance(self, cur_day, est_cl):
        """
        Predict performance of this portfolio assuming that the closing price relatives
        on |cur_day| are |est_cl|.

        :pre: update_allocation has already been called for |cur_day|
        :return: # of dollars that would be made (per dollar of investment) using this
        portfolio if the true closing price relatives on |cur_day| were |est_cl|
        """

        if cur_day == 0:
            # TODO: handle this case better. all portfolios will have same performance on day 0
            # so might as well just return a constant
            return 1
        prev_b = self.b_history[cur_day-1]
        # TODO: case when this portfolio has no money
        est_perf = (1.0 / self.dollars) * self.get_dollars(cur_day, prev_b=prev_b, cur_b=self.b, cpr=est_cl)
        return est_perf

    def print_results(self):
        print 'Total dollar value of assets over time:'
        print self.dollars_history
        print 'Sharpe ratio:'
        print empirical_sharpe_ratio(self.dollars_history)
        #plt.plot(self.dollars_hist)
        #plt.show()
