""""
    Online Moving Average Reversion (OLMAR)
"""

import util
import numpy as np
from math import pow


# TODO: account for 0 stocks!!!!!!

class OLMAR(util.Portfolio):
    def __init__(self, market_data, window=5, eps=10):
        """

        :param market_data: The data to be used
        :param window: Window size (in days)
        :param eps:
        :return:
        """
        if eps <= 1:
            raise Exception('Epsilon must be > 1.')
        if window < 1:
            raise Exception('Window length must be at least 1, and it is recommended that the window be >= 3.')

        self.window = window
        self.eps = eps
        super(OLMAR, self).__init__(market_data)

    def predict_price_relatives(self, tomr):
        """
        It is currently the end of day |tomr|-1.
        This function predicts the price relative vector at the end of |tomr| based on the moving average
        in the window |tomr|-w to |tomr|-1:

        x_t+1 = MovingAvg/p_t = (1/w)(p_t/p_t + p_t-1/p_t + ... + p_t-w+1/p_t)

        :param tomr: The day to predict the closing price relatives for.
        (This plays the role of t+1 in the above equation.)
        :return: The predicted price relatives vector.
        """
        window = self.window
        if tomr <= window:
            # Full window is not available
            window = tomr - 1

        window_cl = self.data.cl[tomr-window:tomr, :]
        moving_avg_cl = np.mean(window_cl, axis=0)
        today_cl = self.data.cl[tomr-1, :]
        price_rel = np.divide(moving_avg_cl, today_cl)
        return price_rel

    def compute_lambda(self, predicted_price_rel, mean_ppl):
        # TODO: check that this is actually L2 norm
        l2_norm = np.linalg.norm(predicted_price_rel - mean_ppl*np.ones(len(predicted_price_rel)))
        predicted_under_eps = self.eps - np.dot(self.allocation, predicted_price_rel)

        # TODO: check!!
        return max(0, predicted_under_eps / (pow(l2_norm, 2)))

    def get_new_allocation(self, day):
        predicted_price_rel = self.predict_price_relatives(day+1)
        mean_price_rel = np.mean(predicted_price_rel)  # x bar at t+1
        lam = self.compute_lambda(predicted_price_rel, mean_price_rel)  # lambda at t+1

        # limit lambda to avoid numerical problems
        # TODO: check if this is necessary!!
        #lam = min(100000, lam)

        # TODO: may need to do simplex projection
        return self.allocation + lam * (predicted_price_rel - mean_price_rel*np.ones(len(predicted_price_rel)))

    def update_portfolio(self, day):
        today_cl = np.nan_to_num(self.data.cl[day, :])
        #total_dollars = util.dollars_in_stocks(self.shares_holding, today_cl)
        yesterday_cl = np.nan_to_num(self.data.cl[day-1, :])
        rel_change = np.nan_to_num(np.divide(today_cl, yesterday_cl))
        total_dollars_change = np.dot(self.allocation, rel_change)
        total_dollars = self.dollars_hist[-1] + total_dollars_change
        self.dollars_hist.append(total_dollars)
        self.allocation = self.get_new_allocation(day)

        # TODO: transaction costs!!!, simplex projection

    def run(self):
        for day in range(1, self.num_days):
            self.update_portfolio(day)
            #self.shares_holding = self.update_portfolio(day)
            #self.shares_held_hist = np.append(self.shares_held_hist, [self.shares_holding], axis=0)
            # TODO: make sure allocation sums to 1
        self.print_results()
        self.save_results()

    def print_results(self):
        print 30 * '-'
        print 'Performance for uniform constant rebalancing portfolio:'
        print 30 * '-'
        #print self.shares_holding
        print 'Total dollar value of assets:'
        print util.dollars_in_stocks(self.shares_holding, self.data.cl[-1, :])
        print 'Sharpe ratio: TODO'
        #plt.plot(self.dollars_hist)
        #plt.show()

    def save_results(self):
        output_fname = 'results/olmar.txt'
        print 'Saving dollar value to file: ', output_fname
        output_file = open(output_fname, 'w')
        output_file.write('\t'.join(map(str, self.dollars_hist)))

