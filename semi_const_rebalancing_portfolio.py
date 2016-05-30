"""
    This file contains the unifrom semi-constant rebalancing portfolio
    class, which inherits from the Portfolio class.

"""

import numpy as np

import util
from constants import cost_per_trans_per_dollar


class UniformSemiConstantRebalancedPortfolio(util.Portfolio):

    def __init__(self, market_data, interval):
        self.interval = interval
        super(UniformSemiConstantRebalancedPortfolio, self).__init__(market_data)

    def get_new_allocation(self, cur_day):
        """

        :param cur_day:
        :return: A (1 x num_stocks) array of fractions. Each fraction represents the
        amount of the money should be invested in that stock at the end of the day.
        """
        stocks_avail = util.get_avail_stocks(self.data.op[cur_day, :])
        num_stocks_avail = len(stocks_avail.keys())
        new_allocation = np.zeros(self.num_stocks)
        for stock in stocks_avail.keys():
            new_allocation[stock] = 1.0 / num_stocks_avail

        return new_allocation

    def update_portfolio(self, cur_day, new_allocation):
        """
        A naive approach. Rebalances the stock holdings so that an equal amount of money is invested
        into each stock.

        :param cur_day:
        :param new_allocation: A (1 x num_stocks) array of fractions. Each fraction represents the
        amount of the money should be invested in that stock at the end of the day.
        :return: List of the new share holdings
        """

        #print 'Reinvesting into a uniform portfolio for day: ', cur_day
        close_prices = self.data.cl[cur_day, :]
        total_dollars = util.dollars_in_stocks(self.shares_holding, close_prices)
        self.dollars_hist.append(total_dollars)

        if cur_day % self.interval == 0:
            # Apply the CRP approach
            num_stocks_to_invest_in = np.count_nonzero(new_allocation)

            dist_from_uniform = util.dollars_away_from_uniform(self.shares_holding, close_prices, 1.0 * total_dollars / num_stocks_to_invest_in)
            total_trans_costs = dist_from_uniform * cost_per_trans_per_dollar
            rebalanced_dollars = 1.0 * (total_dollars - total_trans_costs)/ num_stocks_to_invest_in

            stocks_avail = util.get_avail_stocks(self.data.op[cur_day, :])
            new_share_holdings = np.zeros(self.num_stocks)
            for idx in stocks_avail.keys():
                new_share_holdings[idx] = 1.0 * rebalanced_dollars / close_prices[idx]

            return new_share_holdings
        else:
            # Don't apply the CRP approach on this day
            prev_close_prices = np.nan_to_num(self.data.cl[cur_day-1, :])
            rel_change = np.nan_to_num(np.divide(close_prices, prev_close_prices))
            new_share_holdings = np.multiply(self.shares_holding, rel_change)
            return new_share_holdings

    def run(self):
        for day in range(1, self.num_days):
            self.shares_holding = self.update_portfolio(day, self.get_new_allocation(day))
            self.shares_held_hist = np.append(self.shares_held_hist, [self.shares_holding], axis=0)

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
        output_fname = 'results_low_trans_cost/2_semi_const_rebalancing_dollars_over_time_interval_' + str(self.interval) + '.txt'
        print 'Saving dollar value to file: ', output_fname
        output_file = open(output_fname, 'w')
        output_file.write('\t'.join(map(str, self.dollars_hist)))



