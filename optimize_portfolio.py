'''
    Note that some of the stocks do not have data for the beginning
    of the time period. For example, CFN (the 88th stock name)
    does not have any data until day 827.

'''

import util
from const_rebalancing_portfolio import UniformConstantRebalancedPortfolio
from uniform_buy_and_hold import UniformBuyAndHoldPortfolio

"""
*********************
        Main
*********************
"""
if __name__ == "__main__":

    # Load the training data from MATLAB file
    train_data = util.load_matlab_data('portfolio.mat')

    num_stocks = len(train_data.stock_names)  # Number of stocks in the dataset
    num_train_days = train_data.vol.shape[0]
    print 'Number of stocks in training set: ', num_stocks
    print 'Number of days in training set: ', num_train_days

    const_portfolio = UniformConstantRebalancedPortfolio(market_data=train_data)
    const_portfolio.run()

    ubah_portfolio = UniformBuyAndHoldPortfolio(market_data=train_data)
    ubah_portfolio.run()

