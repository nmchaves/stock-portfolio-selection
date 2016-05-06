'''
    Note that some of the stocks do not have data for the beginning
    of the time period. For example, CFN (the 88th stock name)
    does not have any data until day 827.

'''

import numpy as np
import util

cost_per_trans_per_dollar = 0.0005  # Cost of buying or shorting $1 of stock
init_dollars = 1


class PortfolioOptimizer(util.MarketData):
    def __init__(self, market_data, strategy):
        self.data = market_data
        self.dollars = init_dollars
        self.strategy = strategy
        self.shares_holding = util.init_portfolio_naive(self.data, self.dollars, cost_per_trans_per_dollar)
        self.shares_held_hist = np.empty((0, num_stocks))  # History of share holdings
        self.shares_held_hist = np.append(self.shares_held_hist, self.shares_holding)
        self.num_days = self.data.vol.shape[1]

    def rebalance(self):
        if self.strategy == 'fixed':
            print self.strategy

    def rebalance_naive(self, cur_day):
        """

        :param cur_day:
        :return:
        """
        today_open = self.data.op[cur_day-1, :]
        cur_dollars = np.dot(self.shares_holding, today_open)  # We are NOT allowed to use low, high, and close
                                                        # from today to decide how much to buy at the end of the day
        new_shares = np.zeros(num_stocks)
        for idx, share in enumerate(self.shares_holding):
            # TODO
            new_shares[idx] = share

        return new_shares

    def run(self):
        print self.data.stock_names[0]
        print self.num_days

        for day in range(2, num_train_days+1):
            # TODO
            self.rebalance()
            break

    def print_results(self):
        print 'TOOD: print out summary results...'



"""
*********************
        Main
*********************
"""
if __name__ == "__main__":

    # Load the training data
    train_data = util.load_matlab_data('portfolio.mat')

    num_stocks = len(train_data.stock_names)  # Number of stocks in the dataset
    num_train_days = train_data.vol.shape[0]
    print 'Number of stocks in training set: ', num_stocks
    print 'Number of days in training set: ', num_train_days

    opt = PortfolioOptimizer(market_data=train_data, strategy='fixed')
    opt.run()
