"""
    Utilities file
"""
import numpy as np
from scipy import io
from math import isnan
from constants import init_dollars, cost_per_trans_per_dollar


class Portfolio(object):
    def __init__(self, market_data):
        self.data = market_data
        self.num_stocks = self.data.vol.shape[1]  # Total number of stocks in dataset (not all are available initially)
        self.dollars = init_dollars
        self.allocation, self.shares_holding = init_portfolio_uniform(self.data, self.dollars, cost_per_trans_per_dollar)
        self.shares_held_hist = np.empty([0, self.num_stocks])  # History of share holdings
        self.shares_held_hist = np.append(self.shares_held_hist, [self.shares_holding], axis=0)
        self.num_days = self.data.vol.shape[0]
        self.dollars_hist = [1]

    def update_portfolio(self, cur_day, new_allocation):
        raise('update_portfolio is an abstract method, so it must be implemented by the child class!')


class MarketData():
    def __init__(self, vol, op, lo, hi, cl, stocks):
        self.vol = vol
        self.op = op
        self.lo = lo
        self.hi = hi
        self.cl = cl
        self.stock_names = stocks

def load_matlab_data(file_path):

    mat = io.loadmat('portfolio.mat')
    train_vol = np.array(mat['train_vol'])  # Volume for each stocks on each day
    train_op = np.array(mat['train_op'])
    train_lo = np.array(mat['train_lo'])
    train_hi = np.array(mat['train_hi'])
    train_cl = np.array(mat['train_cl'])
    train_stocks = [name[0] for name in np.array(mat['train_stocks'])[0]]  # Ticker names for all 497 stocks

    return MarketData(train_vol, train_op, train_lo, train_hi, train_cl, train_stocks)


def init_portfolio_uniform(data, dollars, cost_per_trans_per_dollar):
    """
    This function initializes the share holdings by naively investing equal
    amounts of money into each stock.

    :param data: MarketData object containing the data for
    detemrining how to allocate the portfolio

    :returns A list of the allocation (fraction of |dollars| given to each stock) and
    a list of amount of shares allocated to each stock
    """

    available_stocks = get_avail_stocks(data.op[0, :])
    num_stocks_avail = len(available_stocks.keys())

    total_trans_cost = dollars * cost_per_trans_per_dollar
    dollars_per_stock = 1.0 * (dollars - total_trans_cost) / num_stocks_avail
    allocation_per_stock = 1.0 * dollars_per_stock / init_dollars

    print 'Num stocks available at day 1: ', num_stocks_avail
    print 'Dollars allocated to each stock at day 1: ', dollars_per_stock

    # Allocate an equal amount of money to each stock
    num_stocks_total = len(data.stock_names)
    init_allocation = [0] * num_stocks_total
    init_shares = [0] * num_stocks_total
    for index in available_stocks.keys():
        init_allocation[index] = allocation_per_stock
        init_shares[index] = 1.0 * dollars_per_stock / data.cl[0,index]

    return init_allocation, init_shares


def get_avail_stocks(op_prices):
    avail_stocks = {}
    for i, price in enumerate(op_prices):
        # If price is a valid number, then we can purchase the
        # stock at the end of the day
        if not isnan(price):
            avail_stocks[i] = True
    return avail_stocks


def dollars_in_stocks(shares_holding, share_prices):
    # Set any nan values in share_prices to 0
    share_prices = np.nan_to_num(share_prices)

    # Return total amount of money held in stocks
    return np.dot(shares_holding, share_prices)


def dollars_away_from_uniform(shares_holding, share_prices, dollars_per_stock):
    """
    This function determines how far the current portfolio is (in terms of dollars)
    from a uniformly distributed portfolio.

    :param shares_holding: The shares that are currently held
    :param share_prices: The opening share prices today
    :param dollars_per_stock: Number of dollars that would be invested in each stock
    if the portfolio were completely uniform.
    :return: Distance from uniform portfolio in dollars
    """

    distance = 0
    for share, price in zip(shares_holding, share_prices):
        if not isnan(price):
            distance += abs(share * price - dollars_per_stock)

    return distance


def get_float_array_from_file(path):
    """
    This function takes a path to a file containing a tab-delimited
    text file and converts that file into an array.

    :param path: Path to the file
    :return: The array
    """
    f = open(path)
    for line in f:
        return map(float, line.split('\t'))
        break


