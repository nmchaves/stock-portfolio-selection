import numpy as np
from olmar import OLMAR

import util
from portfolio import Portfolio


class RMR(OLMAR):
    """"
    Robust Median Reversion Portfolio. Similar to OLMAR, except that it
    estimates the price relatives using the L1 median.

    Reference:
    Robust Median Reversion Strategy for On-Line Portfolio Selection
    by Huang et al:
    http://www.ijcai.org/Proceedings/13/Papers/296.pdf

    """
    def __init__(self, market_data, window=5, eps=10, tau=0.001, max_iter=100):

        self.tau = tau  # tolerance level
        self.max_iter = max_iter
        super(RMR, self).__init__(market_data, window, eps)

    def predict_price_relatives(self, day):
        """
        This function predicts the price relative vector at the end of |day| based on the L1 median
        in the window |day|-w to |day|-1:


        :param day: The day to predict the closing price relatives for.
        (This plays the role of t+1 in the above equation.)
        :return: The predicted price relatives vector.
        """
        window = self.window
        if day <= window:
            # Full window is not available
            window = day

        window_cl = self.data.get_cl(relative=False)[day-window:day, :]
        today_op = self.data.get_op(relative=False)[day, :]
        today_op = np.reshape(today_op, newshape=(1, self.num_stocks))
        window_prices = np.append(window_cl, today_op, axis=0)

        mu = np.median(window_prices, axis=0)  # TODO: median when window size is 2?? np gives mean...

        for i in range(1, self.max_iter):
            prev_mu = mu
            mu = self.T_func(mu, window_prices)
            L1_dist = np.linalg.norm((prev_mu-mu), ord=1)
            thresh = self.tau * np.linalg.norm(mu, ord=1)
            if L1_dist <= thresh:
                break

        price_rel = util.silent_divide(mu, today_op)
        return price_rel
        """
        window_cl = self.data.get_cl(relative=False)[day-window:day, :] # TODO: check
        prev_cl = self.data.get_cl(relative=False)[day-1, :]

        if window > 1:
            mu = np.median(window_cl, axis=0)  # TODO: median when window size is 2?? np gives mean...
        else:
            return prev_cl  # TODO: cleaner approach. this prevents divide by 0 in T_func


        for i in range(1, self.max_iter):
            prev_mu = mu
            mu = self.T_func(mu, window_cl)
            L1_dist = np.linalg.norm((prev_mu-mu), ord=1)
            thresh = self.tau * np.linalg.norm(mu, ord=1)
            if L1_dist <= thresh:
                break

        price_rel = np.nan_to_num(np.true_divide(mu, prev_cl))
        return price_rel
        """

    def T_func(self, mu, window_cl):
        """
        The function described in Proposition 1 of Huang et al.
        """

        # TODO: is eta always 0 as describe in the paper?? Seems like the
        # price vector will never remain constant over the whole window
        s1 = 0
        s2 = 0
        R_tilde = 0
        for cl in window_cl:
            diff = cl - mu
            is_nonzero = np.any(diff)
            if is_nonzero:
                dist = np.linalg.norm((cl-mu), ord=2)
                s1 += 1.0 / dist
                s2 += np.nan_to_num(np.true_divide(cl, dist))
                R_tilde += diff * 1.0 / dist

        T_bar = s2 * 1.0 / s1

        gamma = np.linalg.norm(R_tilde, ord=2)
        gamma_inv = 1.0 / gamma

        return T_bar + min(1.0, gamma_inv) * mu

    def print_results(self):
        print 30 * '-'
        print 'Performance for RMR:'
        print 30 * '-'
        Portfolio.print_results(self)

    def save_results(self):
        output_fname = 'results/new/rmr.txt'
        util.save_results(output_fname, self.dollars_history)



