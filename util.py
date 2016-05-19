"""
    Utilities file
"""
import numpy as np
from scipy import io
from math import isnan, sqrt
import market_data


def load_matlab_sp500_data(file_path):
    """
    Get raw stock market data from Matlab file in |file_path|

    :param file_path: Path to the data file (must be a .mat file)
    :return: MarketData object containing stock market data.
    """
    mat = io.loadmat(file_path)
    train_vol = np.array(mat['train_vol'])  # Volume for each stocks on each day
    train_op = np.array(mat['train_op'])
    train_lo = np.array(mat['train_lo'])
    train_hi = np.array(mat['train_hi'])
    train_cl = np.array(mat['train_cl'])
    train_stocks = [name[0] for name in np.array(mat['train_stocks'])[0]]  # Ticker names for all 497 stocks

    return market_data.MarketData(train_vol, train_op, train_lo, train_hi, train_cl, train_stocks)


def get_price_relatives(raw_prices):
    """
    Converts raw stock market prices to relative price changes.
    Sets the day 1 price relatives to be 0 by default. # TODO: change this when working with test set!!

    :param raw_prices: (NUM_DAYS x NUM_STOCKS) Array of raw stock market prices
    :return: Array of relative price changes
    """
    price_relatives = np.zeros(raw_prices.shape)
    prev_row = raw_prices[0]
    for (i, row) in enumerate(raw_prices[1:]):
        for (j, price) in enumerate(row):
            prev_price = prev_row[j]
            if price != 0 and prev_price != 0:
                # TODO: check for edge cases
                price_relatives[i+1, j] = 1.0 * price / prev_price
        prev_row = row
    return price_relatives


def get_avail_stocks(op_prices):
    """
    Based on the opening prices for a given day, get the set of stocks that one can
    actually purchase today (not all stocks are on the market at all times)

    :param op_prices: The list of opening prices (this is NOT the relative prices! if you
    use the relative prices, then you may miss the 1st day that a stock is available b/c
    the relative price won't be defined)
    :return: Binary array of available stocks (1 means available, 0 means unavailable
    """
    avail_stocks = [0] * len(op_prices)
    for i, price in enumerate(op_prices):
        # If price is a valid number, then we can purchase the
        # stock at the end of the day
        if not isnan(price):
            avail_stocks[i] = 1
    return avail_stocks


def get_uniform_allocation(num_stocks, op_prices):
    b = np.zeros(num_stocks)
    available_stocks = get_avail_stocks(op_prices)
    num_stocks_avail = np.count_nonzero(available_stocks)
    frac = 1.0 / num_stocks_avail  # fractional allocation per stock

    for (i, is_avail) in enumerate(available_stocks):
        if is_avail:
            b[i] = frac
    return b


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


def emprical_sharpe_ratio(dollars):
    """
    Compute the empirical Sharpe Ratio:
        x_bar = (final_dollars - init_dollars) / num_days
        var = sqrt( (1/num_days) * (sum(x_i-mean(x_bar)))^2 )
        Sharpe ratio = mean(x) * sqrt(num_days) / var

    :param dollars: # of dollars held at the end of each day over time.
    :return: Sharpe ratio
    """

    # TODO: fix this, e.g. determine if dollars[0]=1 or 0.9995 and fix variance computation
    num_days = len(dollars)
    x = [dollars[i] - dollars[i-1] for i in range(1, num_days)]
    print 'x: '
    print x
    x_bar = 1.0 * (dollars[-1] - dollars[0]) / num_days
    #var = sqrt((1.0/num_days) * )
    #return 1.0 * x_bar * sqrt(num_days) / var
