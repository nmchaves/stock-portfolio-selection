import numpy as np
from math import pow
from portfolio import Portfolio
import util
import itertools

# todo: tune eps
# Todo: tuning should only take place over last couple weeks or so...

class OLMAR(Portfolio):
    """"

    Online Moving Average Reversion (OLMAR) Portfolio

    Introduced by Li and Hoi. "On-Line Portfolio Selection with Moving Average Reversion"

    """
    def __init__(self, market_data, start=0, stop=None, window=20, eps=1.1, rebal_interval=1,
                 window_range=range(16, 26, 2), eps_range=[1.1, 1.2, 1.3, 1.4, 1.5], tune_interval=None,
                 init_b=None, verbose=False, silent=False):
        """

        :param market_data: Stock market data (MarketData object)
        :param window: Window size (in days)
        :param eps: Epsilon parameter (the passive vs aggressive threshold). If OLMAR projects that it will
        increase its wealth by more than a factor of |eps|, then it rebalances. Otherwise, it keeps the allocation
        the same.
        :param rebal_interval: Rebalance interval (Rebalance the portfolio every |reb_int| days)
        """
        if eps <= 1:
            raise Exception('Epsilon must be > 1.')
        if window < 1:
            raise Exception('Window length must be at least 1, and it is recommended that the window be >= 3.')

        self.window = window
        self.eps = eps

        # Tuning space for hyperparameters
        self.window_range = window_range
        self.eps_range = eps_range

        super(OLMAR, self).__init__(market_data=market_data, start=start, stop=stop, rebal_interval=rebal_interval,
                                    init_b=init_b, tune_interval=tune_interval, verbose=verbose, silent=silent)

    def predict_price_relatives(self, day):
        """
        This function predicts the price relative vector at the end of |day| based on the moving average
        in the window |day|-w to |day|-1:

        x_t+1 = MovingAvg/p_t = (1/w)(p_t/p_t + p_t-1/p_t + ... + p_t-w+1/p_t)

        TODO: check if this actually makes sense...
        Note: Since we have access to the open prices, we let p_t be the open price on |day|. The other
        price p_t-i are all closing prices.

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
        avg_prices = np.mean(window_prices, axis=0)  # Mean of each stock in the window

        price_rel = util.silent_divide(avg_prices, today_op)  # Predicted price relatives
        return price_rel[0]

    def compute_lambda(self, ppr_avail, mean_ppr, avail_idxs):
        num_avail_stocks = len(ppr_avail)
        l2_norm = np.linalg.norm(ppr_avail - mean_ppr*np.ones(num_avail_stocks), ord=2)

        # TODO: check if something in RMR causes this
        if l2_norm == 0:
            return 0
        avail_b = np.array(self.b)[avail_idxs]  # Current allocations for available stocks
        predicted_under_eps = self.eps - np.dot(avail_b, ppr_avail)

        # TODO: check if we need to simplex project
        return max(0, predicted_under_eps / (pow(l2_norm, 2)))

    def get_new_allocation(self, day, init=False):
        """

        Determine the new desired allocation for the end of |day| using
        the OLMAR algorithm.

        :param day:
        :param init: If True, this portfolio is being initialized today.
        :return:
        """
        if init:
            cur_day_op = self.data.get_op(relative=False)[day, :]  # opening prices on |cur_day|
            return util.get_uniform_allocation(self.num_stocks, cur_day_op)

        predicted_price_rel = self.predict_price_relatives(day)

        # Compute mean price relative of available stocks (x bar at t+1)
        today_op = self.data.get_op(relative=False)[day, :]
        avail_stocks = util.get_avail_stocks(today_op)
        avail_idxs = util.get_available_inds(avail_stocks)
        ppr_avail = predicted_price_rel[avail_idxs]  # predicted price relatives of available stocks
        mean_price_rel = np.mean(ppr_avail)

        lam = self.compute_lambda(ppr_avail, mean_price_rel, avail_idxs)  # lambda at t+1

        # limit lambda to avoid numerical problems from acting too aggressively.
        # (referenced from marigold's implementation: https://github.com/Marigold/universal-portfolios)
        # TODO: check if this is necessary
        lam = min(100000, lam)

        # Note: we don't perform simplex project b/c negative values (shorting) is allowed.
        new_b = np.zeros(self.num_stocks)
        for i, _ in enumerate(new_b):
            ppr = predicted_price_rel[i]
            if ppr > 0:
                new_b[i] = self.b[i] + lam * (ppr - mean_price_rel)

        # Normalize b so that it sums to 1
        sum_b = np.linalg.norm(new_b, ord=1)
        return (1.0 / sum_b) * new_b

    def tune_hyperparams(self, cur_day):
        # Create new instances of this portfolio with various hyperparameter settings
        # to find the best constant hyperparameters in hindsight

        tune_duration = 20
        if cur_day >= tune_duration:
            start_day = cur_day - tune_duration
        else:
            # Not worth tuning yet
            return

        hyperparam_space = [self.window_range, self.eps_range]
        hyp_combos = list(itertools.product(*hyperparam_space))

        #init_b = self.b_history[-tune_duration]  # Allocation used at beginning of tuning period
        init_b = None

        # Compute sharpe ratios for each setting of hyperparams
        sharpe_ratios = []
        for (win, eps) in hyp_combos:
            cur_portfolio = OLMAR(market_data=self.data, start=start_day, stop=cur_day,
                                init_b=init_b, window=win, eps=eps, tune_interval=None, verbose=False, silent=True)
            cur_portfolio.run(start_day, cur_day)
            cur_dollars_history = cur_portfolio.get_dollars_history()
            sharpe_ratios.append(util.empirical_sharpe_ratio(cur_dollars_history))

        best_window, best_eps = hyp_combos[sharpe_ratios.index(max(sharpe_ratios))]
        self.window = best_window
        self.eps = best_eps
        return

    def print_results(self):
        if self.verbose:
            print 30 * '-'
            print 'Performance for OLMAR:'
            print 30 * '-'
        Portfolio.print_results(self)

    def save_results(self):
        output_fname = 'results/new/olmar.txt'
        util.save_results(output_fname, self.dollars_history)

