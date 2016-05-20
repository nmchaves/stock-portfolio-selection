"""
    Class to pool together multiple experts (ie multiple portfolios).

    This pool of experts is itself a portfolio.
"""


from constants import init_dollars
from util import predict_prices
from portfolio import Portfolio
from market_data import MarketData
import numpy as np


class ExpertPool(Portfolio):
    def __init__(self, market_data, experts, weights=None):
        if not isinstance(market_data, MarketData):
            raise 'market_data input to ExpertPool constructor must be a MarketData object.'

        for expert in experts:
            if not isinstance(expert, Portfolio):
                raise '\"experts\" argument to ExpertPool constructor must be a list of Portfolio objects.'

        self.experts = experts
        self.num_experts = len(experts)
        self.data = market_data
        self.num_stocks = len(self.data.stock_names)
        self.num_days = self.data.get_vol().shape[0]
        self.dollars = init_dollars
        self.b_history = []
        self.dollars_history = []

    def aggregate_experts(self, weights):
        net_b = np.zeros(self.num_stocks) #[0.0] * self.num_experts  # weighted sum of expert allocations
        for (idx, expert) in enumerate(self.experts):
            net_b += np.multiply(weights[idx], expert.get_b())
        return net_b

    def get_new_allocation(self, cur_day):
        if cur_day == 0:
            # TODO: enable one to specify an initial weight distribution (eg one might believe some portfolios
            # will be better than others)
            for expert in self.experts:
                expert.update_allocation(cur_day)

            # Use uniform weights for all experts
            weights = (1.0 / self.num_experts) * np.ones(self.num_experts)
            net_b = self.aggregate_experts(weights)
            return net_b

        est_cl = predict_prices(cur_day, self.data)  # Estimate closing prices

        # Predict return per dollar invested into each expert based on estimated prices
        preds = []
        for expert in self.experts:
            expert.update_allocation(cur_day)
            predicted_performance = expert.predict_performance(cur_day=cur_day, est_cl=est_cl)
            preds.append(predicted_performance)

        weights = np.multiply((1.0 / sum(preds)), preds)
        net_b = self.aggregate_experts(weights)
        return net_b

    def run(self):
        for day in range(0, self.num_days):
            self.update(day)
        self.print_results()

    def print_results(self):
        print 30 * '-'
        print 'Results of expert pooling:'
        print 30 * '-'
        Portfolio.print_results(self)
