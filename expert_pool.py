import pdb
from math import exp
import numpy as np
import util
from constants import init_dollars
from market_data import MarketData
from portfolio import Portfolio
from util import predict_prices

# TODO: throw error if some experts receive a history, but other experts don't
# TODO: is there a better approach to normalization of past dollars history?

class ExpertPool(Portfolio):
    """

    Class to pool together multiple experts (ie multiple portfolios).
    Each day, we predict the performance of each expert (e.g., based on the
    predicted stock prices at the end of the day). Then, we distribute
    money to each of the experts according to their predicted performance.

    """

    def __init__(self, market_data, experts, start=0, stop=None, init_weights=None,
                 rebal_interval=1, tune_interval=None,
                 init_b=None, init_dollars=init_dollars,
                 weighting_strategy='exp_window', windows=[10], ew_alpha=0.5, ew_eta=0.1,
                 verbose=False, silent=False, past_results_dir=None, new_results_dir=None, repeat_past=False):

        if not isinstance(market_data, MarketData):
            raise 'market_data input to ExpertPool constructor must be a MarketData object.'

        for expert in experts:
            if not isinstance(expert, Portfolio):
                raise '\"experts\" argument to ExpertPool constructor must be a list of Portfolio objects.'

        self.portfolio_type = 'EP'

        # Check that |weighting_strategy| is valid.
        weighting_strategies = ['exp_window', 'open_price', 'ma_perf']
        if weighting_strategy not in weighting_strategies:
            valid_strats = ', '.join(weighting_strategies)
            raise Exception('Invalid weighting strategy passed to ExpertPool constructor. ' \
                 'Weighting strategy must be 1 of: ' + valid_strats)

        self.experts = experts
        self.num_experts = len(experts)
        self.weighting_strategy = weighting_strategy
        if weighting_strategy == 'exp_window' or weighting_strategy == 'ma_perf':
            self.windows = windows

        # TODO: Check if past history was passed and load in hyperparams (see init of RMR and OLMAR for examples)
        # Note that this is only if we want to actually saw alpha and eta from training, which may be a good idea
        # although it's not a top priority.

        # Parameters for exponential window weighting
        self.ew_alpha = ew_alpha
        self.ew_eta = ew_eta

        super(ExpertPool, self).__init__(market_data, start=start, stop=stop, rebal_interval=rebal_interval,
                                         init_b=init_b, tune_interval=tune_interval, verbose=verbose, silent=silent,
                                         past_results_dir=past_results_dir, new_results_dir=new_results_dir, repeat_past=repeat_past)

    def aggregate_experts(self, weights):
        net_b = np.zeros(self.num_stocks)  # weighted sum of expert allocations (must sum to 1)
        for (idx, expert) in enumerate(self.experts):
            net_b += np.multiply(weights[idx], expert.get_b())

        # Normalize so that b sums to 1 (positive and negative experts will have canceled each other out)
        sum_b = 1.0 * np.linalg.norm(net_b, ord=1)
        return np.true_divide(net_b, sum_b)

    def update_allocation(self, cur_day, init=False):
        # Update the individual experts
        for expert in self.experts:
            expert.update(cur_day, init)

        # Call the regular update_allocation method
        super(ExpertPool, self).update_allocation(cur_day=cur_day, init=init)
        return

    def get_new_allocation(self, cur_day, init=False):
        if self.data_train is None and cur_day < 3:
            # Use uniform weights for all experts, since we have limited info
            # (Need at least 3 days of history to define sharpe ratio)
            weights = (1.0 / self.num_experts) * np.ones(self.num_experts)
        else:
            if self.weighting_strategy == 'open_price':
                weights = self.open_price_weighting(cur_day)
            elif self.weighting_strategy == 'ma_perf':
                weights = self.ma_performance_weighting(cur_day)
            elif self.weighting_strategy == 'exp_window':
                weights = self.recent_sharpe_weighting(cur_day)

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

    def recent_sharpe_weighting(self, cur_day):
        """
        Compute weights of experts based on:
        (1/2)*exp(eta * SR_w1) + (/2)^2 * exp(eta * SR_w2) + ...
        where SR_wi is the empirical sharpe ratio of an expert in window i.

        :param cur_day:
        :return:
        """

        windows = self.windows
        len_windows = sum(windows)

        if cur_day < len_windows:
            expert = self.experts[0]  # if 1 expert has history, then all of them should
            if expert.past_dollars_history is None:
                # Not enough data to use all of the windows and no previous training data available.
                windows = [cur_day]

        # Get the window ranges
        window_ranges = []  # list of (start, stop) tuples for each window
        cur_stop = cur_day
        for (i, window) in enumerate(windows):
            cur_start = cur_stop - windows[i]
            window_ranges.append((cur_start, cur_stop))
            cur_stop = cur_start

        num_experts = self.num_experts
        cum_sharpes = np.zeros(shape=(1, num_experts))  # sharpe ratios of each expert (summed over each window)

        for i, expert in enumerate(self.experts):
            exp_new_dollars_hist = expert.get_dollars_history()
            exp_past_dollars_hist = None
            if expert.past_dollars_history is not None:
                # Get the expert's past history and normalize so that the history ends with the same amount
                # of money that this ExpertPool started with
                exp_past_dollars_hist = expert.past_dollars_history
                final_past = exp_past_dollars_hist[-1]
                exp_past_dollars_hist *= (1.0 * exp_new_dollars_hist[0]) / final_past

            for w, (cur_start, cur_stop) in enumerate(window_ranges):
                if cur_start < 0 and cur_stop <= 0:
                    # Get window from the prior history
                    if exp_past_dollars_hist is None:
                        raise Exception('Invalid window from ' + str(cur_start) + ' to ' + str(cur_stop) +
                                        ' for expert with no prior history available.')
                    else:
                        if cur_stop == 0:
                            window_dollars = exp_past_dollars_hist[cur_start:]
                        else:
                            window_dollars = exp_past_dollars_hist[cur_start:cur_stop]
                elif cur_start < 0 < cur_stop:
                    # Get beginning of window from prior history and the rest from the new history
                    window_dollars_past = exp_past_dollars_hist[cur_start:]
                    window_dollars_new = exp_new_dollars_hist[:cur_stop]
                    window_dollars = np.concatenate([window_dollars_past, window_dollars_new])
                elif cur_start >= 0 and cur_stop > 0:
                    # Get window from the new history
                    window_dollars = exp_new_dollars_hist[cur_start:cur_stop]

                # Perform Exponential Weighting and Scale Down Older Windows
                scale = (1.0 * self.ew_alpha)**w
                cum_sharpes[0, i] += scale * exp(self.ew_eta * util.empirical_sharpe_ratio(window_dollars))

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
            # Note: may want to convert these to relative returns
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

    def save_results(self):
        super(ExpertPool, self).save_results()

        # Save the results of each individual expert
        for expert in self.experts:
            if expert.new_results_dir is not None:
                expert.save_results()

    def get_hyperparams_dict(self):
        hyperparams = {
            'Eta': str(self.ew_eta),
            'Alpha': str(self.ew_alpha)
        }
        return hyperparams
