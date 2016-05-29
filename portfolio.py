import util
from util import get_uniform_allocation, empirical_sharpe_ratio
from constants import init_dollars, cost_per_dollar
from market_data import MarketData
import numpy as np

# TODO: tune distance

class Portfolio(object):
    """
    Superclass for any portfolio optimization algorithm.

    Follows the conventions used in most of portfolio optimization
    literature.
    """

    def __init__(self, market_data, start=0, stop=None, rebal_interval=1, tune_interval=None, tune_length=None,
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
            self.num_days = last_day - self.start

        self.rebal_interval = rebal_interval  # How often to rebalance
        self.tune_interval = tune_interval  # How often to tune hyperparams (if at all)

        self.b = init_b  # b[i] = Fraction of total money allocated to stock i
        self.b_history = np.zeros((self.num_stocks, self.num_days))  # portfolio before open of each day
        #self.b_history = []  # History of allocations over time
        #self.dollars_cl = None  # Dollars at end of current day (after close/trades)
        self.dollars_op_history = np.zeros(self.num_days)
        self.dollars_op_history[0] = init_dollars
        #self.dollars_op_history = [init_dollars]  # Dollars before open each day
        self.dollars_cl_history = np.zeros(self.num_days)  # Dollars before close each day
        self.last_close_price = np.NaN * np.ones(self.num_stocks)
        self.verbose = verbose

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
        if self.tune_interval:
            if cur_day > 0 and cur_day % self.tune_interval == 0:
                self.tune_hyperparams(cur_day)

        self.update_allocation(cur_day, init)
        self.update_dollars(cur_day)

        #new_dollars = self.calc_dollars(cur_day, init)
        #self.dollars_cl = new_dollars
        #self.dollars_cl_history.append(new_dollars)
        #self.dollars_cl_history[cur_day] = new_dollars
        return

    def update_allocation(self, cur_day, init=False):
        """

        :param cur_day:
        :param init: If True, this portfolio is being initialized today.
        :return:
        """

        if init and (self.b is not None):
            # b has already been initialized using initialization argument init_b
            # This may be useful for the test set where we may not want to initialize uniformly.
            #self.b_history.append(self.b)
            self.b_history[:, cur_day+1] = self.b
            return

        if (cur_day % self.rebal_interval) != 0:
            # Don't make any trades today (avoid transaction costs)
            # TODO: need to use special flags to indicate hold when using Yanjun's framework.
            return

        self.b = self.get_new_allocation(cur_day, init)

        #if cur_day <= self.num_days-2:
        #    self.b_history[:, cur_day+1] = self.b
        #self.b_history.append(self.b)
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

        new_portfolio = self.b
        cl = self.data.get_cl(relative=False)[cur_day, :]
        prev_cl = self.last_close_price
        """
        if cur_day > 0:
            prev_cl = self.data.get_cl(relative=False)[cur_day-1, :]
        else:
            prev_cl = np.NaN * np.ones(self.num_stocks)
        """
        op = self.data.get_op(relative=False)[cur_day, :]

        # Get the value of our portfolio at the end of Day t before paying transaction costs
        isActive = np.isfinite(op)
        value_vec = self.dollars_op_history[cur_day] * self.b_history[:, cur_day]
        growth = cl[isActive] / prev_cl[isActive]-1
        growth[np.isnan(growth)] = 0
        revenue_vec = value_vec[isActive] * growth
        value_vec[isActive] = value_vec[isActive] + revenue_vec
        self.dollars_cl_history[cur_day] = self.dollars_op_history[cur_day] + np.sum(revenue_vec)

        # At the end of Day t, we use the close price of day t to adjust our
        # portfolio to the desired percentage.
        if cur_day <= self.num_days-2:
            nonActive = np.logical_not(isActive)
            value_realizable = self.dollars_cl_history[cur_day] - np.sum(value_vec[nonActive])
            new_value_vec, trans_cost = util.rebalance(value_vec[isActive], value_realizable,
                                                       new_portfolio[isActive])

            self.dollars_op_history[cur_day+1] = self.dollars_cl_history[cur_day] - trans_cost
            value_vec[isActive] = new_value_vec
            self.b_history[:, cur_day+1] = value_vec / self.dollars_op_history[cur_day+1]

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

        self.print_results()
        #self.save_results()

    def print_results(self):
        if self.verbose:
            print 'Total dollar value of assets over time:'
            print self.dollars_op_history
            #plt.plot(self.dollars_history)
            #plt.show()

        print 'Sharpe ratio:'
        print empirical_sharpe_ratio(self.dollars_op_history)

    '''
        def init_portfolio_uniform(self):
        """
        This function initializes the allocations by naively investing equal
        amounts of money into each stock.
        """
        day_1_op = self.data.get_op(relative=False)[0, :]  # raw opening prices on 1st day
        return get_uniform_allocation(self.num_stocks, day_1_op)
    '''



    """
    Getters
    """
    def get_b(self):
        return self.b

    def get_dollars_history(self):
        return self.dollars_op_history

    '''
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

        cur_dollars = self.dollars  # Dollars before trading
        if cur_day == 0 or init:
            # Only buy stocks on day 0 (no selling)
            trans_costs = self.dollars * cost_per_dollar
            return cur_dollars - trans_costs
        else:
            """
            _, trans_costs = util.rebalance(value_vec[isActive], cur_dollars,
                                            new_portfolio[isActive])
            return cur_dollars - trans_costs
            """

            prev_b = self.b_history[-1]
            cur_b = self.b
            #cpr = self.data.get_cl(relative=True)[cur_day, :]
            cl = self.data.get_cl(relative=False)[cur_day, :]
            prev_cl = self.data.get_cl(relative=False)[cur_day-1, :]
            cur_op = self.data.get_op(relative=False)[cur_day, :]
            return util.get_dollars(cur_day=cur_day, op=cur_op, cl=cl, prev_cl=prev_cl,
                                    cur_b=cur_b, cur_dollars=self.dollars)
            #return util.get_dollars(cur_day, cur_dollars, prev_b, cur_b, cpr)

    '''