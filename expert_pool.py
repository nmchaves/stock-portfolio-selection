from constants import init_dollars
from util import predict_prices
from portfolio import Portfolio
from market_data import MarketData
import numpy as np

# TODO: enable one to specify an initial weight distribution (eg one might believe some portfolios
# will be better than others)


class ExpertPool(Portfolio):
    """

    Class to pool together multiple experts (ie multiple portfolios).
    Each day, we predict the performance of each expert based on the
    predicted stock prices at the end of the day. Then, we distribute
    money to each of the experts according to their predicted performance.

    """

    def __init__(self, market_data, experts, weights=None, weighting_strategy='open_price', ma_perf_window=10):
        if not isinstance(market_data, MarketData):
            raise 'market_data input to ExpertPool constructor must be a MarketData object.'

        for expert in experts:
            if not isinstance(expert, Portfolio):
                raise '\"experts\" argument to ExpertPool constructor must be a list of Portfolio objects.'

        """
        Meaning of various weighting strategies:
        --open_price: Predict that the closing price will be the same as today's opening price. Allocate
        the portfolio according to how well each expert performs under this scenario.

        --ma_perf: Allocate money to each expert based on a moving avg of its recent returns. Relies on the assumption
        that the strategies that have been working well recently will probably continue to work, perhaps b/c
        these strategies are well-suited to the current market
        """
        weighting_strategies = ['open_price', 'ma_perf']

        if weighting_strategy not in weighting_strategies:
            valid_strats = ', '.join(weighting_strategies)
            raise('Invalid weighting strategy passed to ExpertPool constructor. Weighting strategy must be 1 of: ' +
                  valid_strats)

        self.experts = experts
        self.num_experts = len(experts)
        self.weighting_strategy = weighting_strategy
        if weighting_strategy == 'ma_perf':
            self.ma_perf_window = ma_perf_window

        self.data = market_data
        self.num_stocks = len(self.data.stock_names)
        self.num_days = self.data.get_vol().shape[0]
        self.dollars = init_dollars
        self.b_history = []
        self.dollars_history = [self.dollars]

    def aggregate_experts(self, weights):
        net_b = np.zeros(self.num_stocks)  # weighted sum of expert allocations (must sum to 1)
        for (idx, expert) in enumerate(self.experts):
            net_b += np.multiply(weights[idx], expert.get_b())
        return net_b

    def get_new_allocation(self, cur_day):
        if cur_day == 0:
            for expert in self.experts:
                expert.update_allocation(cur_day)

            # Use uniform weights for all experts, since we have little info
            weights = (1.0 / self.num_experts) * np.ones(self.num_experts)
            net_b = self.aggregate_experts(weights)
            return net_b

        if self.weighting_strategy == 'open_price':
            weights = self.open_price_weighting(cur_day)
        elif self.weighting_strategy == 'ma_perf':
            weights = self.ma_performance_weighting(cur_day)

        net_b = self.aggregate_experts(weights)
        return net_b

    def open_price_weighting(self, cur_day):
        """
        Predict that the closing price will be the same as today's opening price. Allocate
        the portfolio according to how well each expert performs under this scenario.

        :param cur_day: Current day
        :return: Fraction of wealth we'll give to each expert for trading at the end of |cur_day|
        """
        est_cl = predict_prices(cur_day, self.data)  # Estimate closing prices

        # Predict return per dollar invested into each expert based on estimated prices
        preds = []
        for expert in self.experts:
            expert.update_allocation(cur_day)
            predicted_performance = expert.predict_performance(cur_day=cur_day, est_cl=est_cl)
            preds.append(predicted_performance)

        weights = np.multiply((1.0 / sum(preds)), preds)
        return weights

    def ma_performance_weighting(self, cur_day):
        """
        Allocate money to each expert based on its recent performance based on a moving avg of returns.

        :param cur_day: Current day
        :return: Fraction of wealth we'll give to each expert for trading at the end of |cur_day|
        """
        window = self.ma_perf_window
        if cur_day <= window:
            # Full window is not available
            window = cur_day

        # TODO: check this code
        ma_returns = []
        for expert in self.experts:
            dollars_history = expert.get_dollars_history()
            cur_dollars = dollars_history[-1]
            window_start_dollars = dollars_history[-(window+1)]
            avg_return_per_day = (cur_dollars - window_start_dollars) * (1.0 / window)
            ma_returns.append(avg_return_per_day)

        weights = (1.0 / sum(ma_returns)) * ma_returns  # Normalize to obtain weights
        return weights

    def run(self):
        for day in range(0, self.num_days):
            self.update(day)
        self.print_results()

    def print_results(self):
        print 30 * '-'
        print 'Results of expert pooling:'
        print 30 * '-'
        Portfolio.print_results(self)
