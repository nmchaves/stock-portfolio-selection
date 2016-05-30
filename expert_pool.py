from constants import init_dollars
from util import predict_prices
from portfolio import Portfolio
from market_data import MarketData
import numpy as np
import util
from math import exp
import pdb

# TODO: enable one to specify an initial weight distribution (eg one might believe some portfolios
# will be better than others)

class ExpertPool(Portfolio):
    """

    Class to pool together multiple experts (ie multiple portfolios).
    Each day, we predict the performance of each expert (e.g., based on the
    predicted stock prices at the end of the day). Then, we distribute
    money to each of the experts according to their predicted performance.

    """

    def __init__(self, market_data, experts, start=0, stop=None, init_weights=None,
                 rebal_interval=1, tune_interval=None,
                 init_b=None, init_dollars=init_dollars, init_dollars_hist=None,
                 weighting_strategy='exp_window', windows=[10], ew_alpha=0.5, ew_eta = 1, saved_results=False, saved_b = None, dollars_hist=None):

        if not isinstance(market_data, MarketData):
            raise 'market_data input to ExpertPool constructor must be a MarketData object.'

        for expert in experts:
            if not isinstance(expert, Portfolio):
                raise '\"experts\" argument to ExpertPool constructor must be a list of Portfolio objects.'

        weighting_strategies = ['exp_window', 'open_price', 'ma_perf']
        """
        Meaning of various weighting strategies:
        --open_price: Predict that the closing price will be the same as today's opening price. Allocate
        the portfolio according to how well each expert performs under this scenario.

        --ma_perf: Allocate money to each expert based on a moving avg of its recent returns. Relies on the assumption
        that the strategies that have been working well recently will probably continue to work, perhaps b/c
        these strategies are well-suited to the current market

        --exp_window:

        """

        if weighting_strategy not in weighting_strategies:
            valid_strats = ', '.join(weighting_strategies)
            raise Exception('Invalid weighting strategy passed to ExpertPool constructor. ' \
                 'Weighting strategy must be 1 of: ' + valid_strats)

        self.rebal_int = 1
        self.experts = experts
        self.num_experts = len(experts)
        self.weighting_strategy = weighting_strategy
        if weighting_strategy == 'exp_window' or weighting_strategy == 'ma_perf':
            self.windows = windows

        # Parameters for exponential window weighting
        self.ew_alpha = ew_alpha
        self.ew_eta = ew_eta

        self.saved_results = saved_results
        self.saved_b = saved_b
        self.saved_dollars = dollars_hist

        super(ExpertPool, self).__init__(market_data, start=start, stop=stop,
                                         rebal_interval=rebal_interval, tune_interval=tune_interval)

    def aggregate_experts(self, weights):
        net_b = np.zeros(self.num_stocks)  # weighted sum of expert allocations (must sum to 1)
        for (idx, expert) in enumerate(self.experts):
            net_b += np.multiply(weights[idx], expert.get_b())

        # Normalize so that b sums to 1 (positive and negative experts will have canceled each other out)
        sum_b = 1.0 * np.linalg.norm(net_b, ord=1)
        return np.true_divide(net_b, sum_b)

    def get_new_allocation(self, cur_day, init=False):
        if cur_day < 3:
            # Use uniform weights for all experts, since we have limited info
            # (Need at least 3 days of history to define sharpe ratio)
            weights = (1.0 / self.num_experts) * np.ones(self.num_experts)
        else:
            if self.weighting_strategy == 'open_price':
                weights = self.open_price_weighting(cur_day)
            elif self.weighting_strategy == 'ma_perf':
                weights = self.ma_performance_weighting(cur_day)
            elif self.weighting_strategy == 'exp_window':
                len_history = cur_day - self.start
                experts_dollars_history = np.zeros(shape=(self.num_experts, len_history))
                for i, expert in enumerate(self.experts):
                    experts_dollars_history[i, :] = expert.get_dollars_history()[0:cur_day]
                weights = self.recent_sharpe_weighting(cur_day, experts_dollars_history)

        # Update the individual experts
        for expert in self.experts:
            expert.update(cur_day, init)

        net_b = self.aggregate_experts(weights)
        return net_b

    def open_price_weighting(self, cur_day):
        """
        Predict that the closing price will be the same as today's opening price. Allocate
        the portfolio according to how well each expert performs under this scenario.

        :param cur_day: Current day
        :return: Fraction of wealth we'll give to each expert for trading at the end of |cur_day|
        """
        est_cl_rel = predict_prices(cur_day, self.data)  # Estimate closing price relatives

        # Predict return per dollar invested into each expert based on estimated prices
        preds = []
        for expert in self.experts:
            expert.update_allocation(cur_day)
            predicted_performance = expert.predict_performance(cur_day=cur_day, est_cl=est_cl_rel)
            preds.append(predicted_performance)

        weights = np.multiply((1.0 / sum(preds)), preds)
        return weights

    def recent_sharpe_weighting(self, cur_day, experts_dollars_history):
        """
        Compute weights of experts based on:
        (1/2)*exp(eta * SR_w1) + (/2)^2 * exp(eta * SR_w2) + ...
        where SR_wi is the empirical sharpe ratio of an expert in window i.

        :param cur_day:
        :return:
        """

        num_experts = experts_dollars_history.shape[0]

        windows = self.windows
        len_windows = sum(windows)

        if cur_day < len_windows:
            # We don't have enough data to use all of the windows
            last_full_win = -1
            for (i, window) in enumerate(windows):
                if cur_day <= sum(windows[0:i+1]):
                    break
                else:
                    last_full_win = i

            if last_full_win >= 0:
                windows = windows[0:last_full_win+1]
            else:
                windows = [cur_day]

        cum_sharpes = np.zeros(shape=(1, num_experts))  # sharpe ratios of each expert (summed over each window)
        for i, dollars_history in enumerate(experts_dollars_history):
            prev_window_start = 0
            for (w, window) in enumerate(windows):
                window_start = cur_day - (window + prev_window_start)
                if w == 0:
                    window_stop = cur_day
                else:
                    window_stop = prev_window_start
                window_dollars = dollars_history[window_start:window_stop]

                # Perform Exponential Weighting and Scale Down Older Windows
                scale = (1.0 * self.ew_alpha)**w
                cum_sharpes[0, i] += scale * exp(self.ew_eta * util.empirical_sharpe_ratio(window_dollars))
                prev_window_start = window_start

        # Normalize to obtain weights that sum to 1
        weights = (1.0 / np.sum(cum_sharpes)) * cum_sharpes
        return weights[0]  # return the array as a single row

    def ma_performance_weighting(self, cur_day):
        """
        Allocate money to each expert based on its recent performance based on a moving avg of returns.

        :param cur_day: Current day
        :return: Fraction of wealth we'll give to each expert for trading at the end of |cur_day|
        """
        window = self.windows[0]
        if cur_day <= window:
            # Full window is not available
            window = cur_day

        ma_returns = np.zeros(shape=(1, self.num_experts))
        for i, expert in enumerate(self.experts):
            dollars_history = expert.get_dollars_history()
            cur_dollars = dollars_history[-1]
            window_dollars = dollars_history[-(window+1):cur_day]

            window_start_dollars = dollars_history[-(window+1)]
            # TODO: should probably convert these to relative returns
            avg_return_per_day = (cur_dollars - window_start_dollars) * (1.0 / window)
            ma_returns[0, i] = avg_return_per_day



        weights = (1.0 / sum(ma_returns)) * ma_returns  # Normalize to obtain weights
        return weights[0]  # return the array as a single row

    def print_results(self):
        print 30 * '-'
        print 'Results of expert pooling: '
        print 30 * '-'
        print 'Weighting strategy: ', self.weighting_strategy
        Portfolio.print_results(self)

    def run(self, start=None, stop=None):
        if start is None:
            start = self.start
        if stop is None:
            stop = self.stop

        if self.saved_results:
            portfolios = self.saved_b
            dollars_history = self.saved_dollars

            weights = None
            for day in range(start, stop):
                # Note: saved_results should be an np array of dimension num stocks x num days to use
                b = portfolios[:,:,day]
                if(day > 3):
                    weights = self.recent_sharpe_weighting(cur_day=day, experts_dollars_history=dollars_history)
                else:
                    weights = (1.0 / self.num_experts) * np.ones(self.num_experts)
                net_b = np.zeros(self.num_stocks)
                for i,experts in enumerate(self.experts):
                    net_b += np.multiply(weights[i], b[i,:])

                sum_b = 1.0 * np.linalg.norm(net_b, ord=1)
                self.b = np.true_divide(net_b, sum_b)

                self.update_dollars(day)
                pdb.set_trace()
        else:
            Portfolio.run(self, start, stop)

