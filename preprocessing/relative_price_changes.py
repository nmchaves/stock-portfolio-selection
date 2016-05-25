"""
    This file converts raw stock market prices to relative price changes.
    The current literature tends to used relative price changes, so we
    adopt this convention as well.
"""

"""
*********************
        Main
*********************
"""

import util

if __name__ == "__main__":
    raw_prices_fname = '../data/portfolio.mat'
    raw_market_data = util.load_matlab_sp500_data(raw_prices_fname)
    price_relatives_dict = raw_market_data.get_price_relatives_dict()

    out_fname = '../data/portfolio_price_relatives.mat'


