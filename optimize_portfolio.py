'''
    Note that some of the stocks do not have data for the beginning
    of the time period. For example, CFN (the 88th stock name)
    does not have any data until day 827.

'''

import numpy as np
import util
from uniform_portfolio import ConstantRebalancedPortfolio


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

    const_portfolio = ConstantRebalancedPortfolio(market_data=train_data)
    const_portfolio.run()
