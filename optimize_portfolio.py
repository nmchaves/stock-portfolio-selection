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
#from nonparametric_markowitz import NonParametricMarkowitz
import numpy as np
import pdb

# TODO: write out tune interval to saved files

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

    """
    olmar = OLMAR(market_data=train_data, tune_interval=None, verbose=True)
    olmar.run()
    """
    olmar2 = OLMAR(market_data=train_data, tune_interval=200, verbose=True, past_results_dir='train_results/OLMAR/')
    olmar2.run()
    #olmar2 = OLMAR(market_data=train_data, tune_interval=10, verbose=True, new_results_dir='train_results/OLMAR/')
    #olmar2.run()
    #rmr = RMR(market_data=train_data, tune_interval=200, verbose=True, new_results_dir='train_results/RMR/')
    #rmr.run()
    """
    olmar = OLMAR(market_data=train_data, tune_interval=None, verbose=True)
    olmar.run()

    rmr2 = RMR(market_data=train_data, tune_interval=100, verbose=True)
    """
    """
    num_experts = 2
    olmar2 = OLMAR(market_data=train_data, tune_interval=None, verbose=True)
    npm = NonParametricMarkowitz(market_data=train_data,  window_len=10, k=10, risk_aversion=1e-5, start_date=25, tune_interval=None, verbose=True)
    
    olmar_b = np.load('olmar_b_hist.npy')
    olmar_dollars = np.load('olmar_dollar_hist.npy')
    npm_b = np.load('npm_b_hist.npy')
    npm_dollars = np.load('npm_dollar_hist.npy')

    b_hist = np.zeros((num_experts, num_stocks, num_train_days))
    b_hist[0] = olmar_b
    b_hist[1] = npm_b
    dollars_history = np.zeros((num_experts, num_train_days))
    dollars_history[0] = olmar_dollars
    dollars_history[1] = npm_dollars

    pool = ExpertPool(market_data=train_data, experts=[olmar2, npm], weighting_strategy='exp_window', ew_eta=0.1, windows=[10], saved_results=True, saved_b = b_hist, dollars_hist=dollars_history)
    pool.run()
    pdb.set_trace()
    """
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

