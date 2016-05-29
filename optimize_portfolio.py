"""
    Use this file for testing out different portfolio optimization strategies

    Note that some of the stocks do not have data for the beginning
    of the time period. For example, CFN (the 88th stock name)
    does not have any data until day 827.

"""

from ucrp import UniformConstantRebalancedPortfolio
from ubah import UniformBuyAndHoldPortfolio
from util import load_matlab_sp500_data
from expert_pool import ExpertPool
from olmar import OLMAR
from rmr import RMR


"""
TODO:
fix handling of negative b's
look into explicitly shorting
tuning parameters, CV. especially for RMR

ARIMA for predicting performance

cross-validation for hyperparameters (e.g. window size)

"""
"""
*********************
        Main
*********************
"""
if __name__ == "__main__":

    # Load the training data from MATLAB file
    train_data = load_matlab_sp500_data('data/portfolio.mat')

    num_stocks = len(train_data.stock_names)  # Number of stocks in the dataset
    num_train_days = train_data.raw['vol'].shape[0]
    print 'Number of stocks in training set: ', num_stocks
    print 'Number of days in training set: ', num_train_days

    print 'No Tuning'
    olmar = OLMAR(market_data=train_data, tune_interval=10, verbose=True)
    olmar.run()
    """
    for tune_int in range(100, 10, -10):
        print 'Tuning every ', str(tune_int), ' days'
        olmar = OLMAR(market_data=train_data, tune_interval=tune_int, verbose=True)
        olmar.run()
    """
    """
    # Expert Pooling using Exponential Window Performance
    rmr3 = RMR(market_data=train_data)
    olmar3 = OLMAR(market_data=train_data)
    #ucrp3 = UniformConstantRebalancedPortfolio(market_data=train_data)
    #ubah3 = UniformBuyAndHoldPortfolio(market_data=train_data)
    pool = ExpertPool(market_data=train_data, experts=[rmr3, olmar3], weighting_strategy='exp_window', windows=[5])
    pool.run()
    """

    """
    rmr = RMR(market_data=train_data)
    rmr.run()
    """
    """
    # Expert Pooling using Moving Average Performance
    rmr3 = RMR(market_data=train_data)
    ucrp3 = UniformConstantRebalancedPortfolio(market_data=train_data)
    ubah3 = UniformBuyAndHoldPortfolio(market_data=train_data)
    olmar3 = OLMAR(market_data=train_data)
    pool = ExpertPool(market_data=train_data, experts=[rmr3, olmar3], weighting_strategy='exp_window', windows=[5])
    pool.run()
    """

    """
    ucrp = UniformConstantRebalancedPortfolio(market_data=train_data)
    ucrp.run()

    ubah = UniformBuyAndHoldPortfolio(market_data=train_data)
    ubah.run()

    rmr = RMR(market_data=train_data)
    rmr.run()

    olmar4 = OLMAR(market_data=train_data)
    rmr4 = RMR(market_data=train_data)
    pool = ExpertPool(market_data=train_data, experts=[olmar4, rmr4], weighting_strategy='open_price')
    pool.run()


    # Expert Pooling using Open Prices
    rmr2 = RMR(market_data=train_data)
    ucrp2 = UniformConstantRebalancedPortfolio(market_data=train_data)
    ubah2 = UniformBuyAndHoldPortfolio(market_data=train_data)
    olmar2 = OLMAR(market_data=train_data)
    pool = ExpertPool(market_data=train_data, experts=[ucrp2, ubah2, olmar2, rmr2], weighting_strategy='open_price')
    pool.run()
    """
