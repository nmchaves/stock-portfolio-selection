import util
from util import get_uniform_allocation, empirical_sharpe_ratio
from constants import init_dollars, cost_per_dollar
from market_data import MarketData
import numpy as np

class Portfolio(object):
    """
    Superclass for any portfolio optimization algorithm.

    Follows the conventions used in most of portfolio optimization
    literature.
    """

    def __init__(self, market_data, start=0, stop=None, rebal_interval=1, tune_interval=None,
                 init_b=None, init_dollars=init_dollars, init_dollars_hist=None, verbose=False):
        """
        :param market_data: Stock market data (MarketData object)
        :param start: What day this portfolio starts at
        :param init_b: An initial allocation to make at the end of the 1st day. This is useful if
        you want to inject prior knowledge about which stocks you think will perform well. Also useful
        for tuning hyperparameters, because we may want the portfolio to start out in a particular state.
        :param rebal_interval: Rebalance interval (Rebalance the portfolio every |reb_int| days)
        """

        if not isinstance(market_data, MarketData):
            raise 'market_data input to Portfolio constructor must be a MarketData object.'

        self.data = market_data
        self.num_stocks = len(self.data.stock_names)
        self.start = start

        if stop:
            self.stop = stop
            self.num_days = stop - start
        else:
            last_day = self.data.get_vol().shape[0]
            self.stop = last_day
            self.num_days = self.start - last_day

        self.rebal_interval = rebal_interval  # How often to rebalance
        self.tune_interval = tune_interval  # How often to tune hyperparams (if at all)

        self.b = init_b  # b[i] = Fraction of total money allocated to stock i
        self.b_history = []  # History of allocations over time
        self.dollars = init_dollars
        self.dollars_history = [self.dollars]
        self.verbose = verbose

    """
    def run(self, start, end):
        raise 'run is an abstract method, so it must be implemented by the child class!'
    """

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

        # Check if we need to tune hyperparameters
        if self.tune_interval:
            if cur_day > 0 and cur_day % self.tune_interval == 0:
                self.tune_hyperparams(cur_day)

        self.update_allocation(cur_day, init)

        new_dollars = self.calc_dollars(cur_day, init)
        self.dollars = new_dollars
        self.dollars_history.append(new_dollars)
        return

    def update_allocation(self, cur_day, init=False):
        """

        :param cur_day:
        :param init: If True, this portfolio is being initialized today.
        :return:
        """

        if self.b is not None:
            self.b_history.append(self.b)

        if init and (self.b is not None):
            # b has already been initialized using initialization argument init_b
            return

        if self.rebal_interval and (cur_day % self.rebal_interval) != 0:
            # Don't make any trades today (avoid transaction costs)
            # TODO: need to use special flags to indicate hold when using Yanjun's framework.
            return

        self.b = self.get_new_allocation(cur_day, init)

    def get_new_allocation(self, cur_day, init=False):
        raise 'get_new_allocation is an abstract method, so it must be implemented by the child class!'

    def init_portfolio_uniform(self):
        """
        This function initializes the allocations by naively investing equal
        amounts of money into each stock.
        """
        day_1_op = self.data.get_op(relative=False)[0, :]  # raw opening prices on 1st day
        return get_uniform_allocation(self.num_stocks, day_1_op)

    def calc_dollars(self, cur_day, init=False):
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
        '''
        if cpr is None:
            cpr = self.data.get_cl(relative=True)[cur_day, :]  # closing price relatives
        if cur_b is None:
            cur_b = self.b
        if prev_b is None and cur_day > 0:
            prev_b = self.b_history[cur_day-1]
        '''
        cur_dollars = self.dollars
        if cur_day == 0 or init:
            # Only buy stocks on day 0 (no selling)
            trans_costs = self.dollars * cost_per_dollar
            return cur_dollars - trans_costs
        else:
            cur_b = self.b
            cpr = self.data.get_cl(relative=True)[cur_day, :]
            prev_b = self.b_history[-1]
            return util.get_dollars(cur_day, cur_dollars, prev_b, cur_b, cpr)

    '''
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
        est_perf = (1.0 / self.dollars) * self.calc_dollars(cur_day, prev_b=prev_b, cur_b=self.b, cpr=est_cl)
        return est_perf
    '''

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

        if self.verbose:
            self.print_results()
            print np.array(self.b_history).shape
        #self.save_results()

    def print_results(self):
        print 'Total dollar value of assets over time:'
        print self.dollars_history
        print 'Sharpe ratio:'
        print empirical_sharpe_ratio(self.dollars_history)
        #plt.plot(self.dollars_hist)
        #plt.show()

    """
    Getters
    """
    def get_b(self):
        return self.b

    def get_dollars_history(self):
        return self.dollars_history
