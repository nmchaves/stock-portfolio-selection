'''
    Utilities file
'''
import numpy as np
from scipy import io
from math import isnan

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


def init_portfolio_naive(data, dollars, cost_per_trans_per_dollar):
    """
    This function initializes the share holdings.

    :param data: MarketData object containing the data for
    detemrining how to allocate the portfolio

    :returns List of amount of shares allocated to each stock
    """

    available_stocks = {}
    for i, price in enumerate(data.op[0,:]):
        # If vol is a valid number, then we can purchase it
        # at the end of day 1
        if not isnan(price):
            available_stocks[i] = True
    num_stocks_avail = len(available_stocks.keys())

    total_trans_cost = dollars * cost_per_trans_per_dollar
    dollars_per_stock = 1.0 * (dollars - total_trans_cost) / num_stocks_avail

    print 'Num stocks available at day 1: ', num_stocks_avail
    print 'Dollars allocated to each stock at day 1: ', dollars_per_stock

    # Allocate an equal amount of money to each stock
    init_shares = np.zeros(len(data.stock_names))
    for index in available_stocks.keys():
        init_shares[index] = 1.0 * dollars_per_stock / data.cl[0,index]

    return init_shares
