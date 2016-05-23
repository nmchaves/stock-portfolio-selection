import numpy as np
from math import pow
from portfolio import Portfolio
import util

# TODO: account for 0 stocks!!!!!!
# TODO: optimal window size?

class OLMAR(Portfolio):
    """"
    Online Moving Average Reversion (OLMAR) Portfolio


    """
    def __init__(self, market_data, window=5, eps=10):
        """

        :param market_data: The data to be used
        :param window: Window size (in days)
        :param eps: Epsilon parameter
        """
        if eps <= 1:
            raise Exception('Epsilon must be > 1.')
        if window < 1:
            raise Exception('Window length must be at least 1, and it is recommended that the window be >= 3.')

        self.window = window
        self.eps = eps
        super(OLMAR, self).__init__(market_data)

    def predict_price_relatives(self, day):
        """
        This function predicts the price relative vector at the end of |day| based on the moving average
        in the window |day|-w to |day|-1:

        x_t+1 = MovingAvg/p_t = (1/w)(p_t/p_t + p_t-1/p_t + ... + p_t-w+1/p_t)

        :param day: The day to predict the closing price relatives for.
        (This plays the role of t+1 in the above equation.)
        :return: The predicted price relatives vector.
        """
        window = self.window
        if day <= window:
            # Full window is not available
            window = day

        # TODO: consider using more than just close prices, since we also have opens!
        window_cl = self.data.get_cl(relative=False)[day-window:day, :]
        moving_avg_cl = np.mean(window_cl, axis=0)
        prev_cl = self.data.get_cl(relative=False)[day-1, :]
        price_rel = np.nan_to_num(np.true_divide(moving_avg_cl, prev_cl))
        return price_rel

    def compute_lambda(self, predicted_price_rel, mean_ppl):
        # TODO: check that this is actually L2 norm
        l2_norm = np.linalg.norm(predicted_price_rel - mean_ppl*np.ones(len(predicted_price_rel)))
        predicted_under_eps = self.eps - np.dot(self.b, predicted_price_rel)

        # TODO: check!! may need to simplex project
        return max(0, predicted_under_eps / (pow(l2_norm, 2)))

    def get_new_allocation(self, day):
        if day == 0:
            cur_day_op = self.data.get_op(relative=False)[day, :]  # opening prices on |cur_day|
            return util.get_uniform_allocation(self.num_stocks, cur_day_op)

        predicted_price_rel = self.predict_price_relatives(day)
        mean_price_rel = np.mean(predicted_price_rel)  # x bar at t+1. TODO: consider if some price rels are 0
        lam = self.compute_lambda(predicted_price_rel, mean_price_rel)  # lambda at t+1

        # limit lambda to avoid numerical problems
        # TODO: check if this is necessary!!
        #lam = min(100000, lam)

        # TODO: need to do simplex projection to handle negatives
        new_b = np.zeros(self.num_stocks)
        for i, _ in enumerate(new_b):
            ppr = predicted_price_rel[i]
            if ppr != 0:
                new_b[i] = self.b[i] + lam * (ppr - mean_price_rel)
        new_b = (1.0/sum(new_b)) * new_b  # normalize b so it sums to 1
        return new_b

    def run(self):
        for day in range(0, self.num_days):
            self.update(day)
            # self.update_portfolio(day)
            # TODO: make sure allocation sums to 1
        self.print_results()
        #self.save_results()

    def print_results(self):
        print 30 * '-'
        print 'Performance for OLMAR:'
        print 30 * '-'
        Portfolio.print_results(self)

    def save_results(self):
        output_fname = 'results/olmar.txt'
        print 'Saving dollar value to file: ', output_fname
        output_file = open(output_fname, 'w')
        output_file.write('\t'.join(map(str, self.dollars_hist)))

