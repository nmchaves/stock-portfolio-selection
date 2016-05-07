
import util
from constants import init_dollars, cost_per_trans_per_dollar
import numpy as np
import matplotlib.pyplot as plt

class UniformBuyAndHoldPortfolio(util.Portfolio):

    def get_new_allocation(self, cur_day):
        """

        :param cur_day:
        :return: A (1 x num_stocks) array of fractions. Each fraction represents the
        amount of the money should be invested in that stock at the end of the day.
        """
        return self.allocation

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

        num_stocks_to_invest_in = np.count_nonzero(new_allocation)
        close_prices = np.nan_to_num(self.data.cl[cur_day, :])
        total_dollars = util.dollars_in_stocks(self.shares_holding, close_prices)
        self.dollars_hist.append(total_dollars)

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
        print 'Performance for uniform buy and hold strategy:'
        print 30 * '-'
        #print self.shares_holding
        print 'Total dollar value of assets:'
        print util.dollars_in_stocks(self.shares_holding, self.data.cl[-1, :])
        print 'Sharpe ratio: TODO'
        #plt.plot(self.dollars_hist)
        #plt.show()

    def save_results(self):
        output_fname = 'results/uniform_buy_and_hold_dollars_over_time.txt'
        print 'Saving dollar value to file: ', output_fname
        output_file = open(output_fname, 'w')
        output_file.write('\t'.join(map(str, self.dollars_hist)))
