"""
    This file contains the unifrom constant rebalancing portfolio
    class, which inherits from the Portfolio class.

    The uniform constant rebalancing approach was implemented as a baseline
    to compare with more sophisticated methods.
"""

import util
from portfolio import Portfolio


class UniformConstantRebalancedPortfolio(Portfolio):

    def update(self, cur_day):
        """
        A naive approach. Rebalances the stock holdings so that an equal amount of money is invested
        into each stock.

        :param cur_day: 0-based index of today
        :return: A (1 x num_stocks) array of fractions. Each fraction represents the
        amount of the money that should be invested in that stock at the end of the day.
        """
        cur_day_op = self.data.raw['op'][cur_day, :]  # opening prices on |cur_day|
        new_b = util.get_uniform_allocation(self.num_stocks, cur_day_op)
        self.b = new_b
        self.b_history.append(new_b)

    def run(self):
        print 'Running UCRP'
        for day in range(1, self.num_days):
            # Note that day 0 allocation is handled in __init__ of Portfolio superclass
            self.update(day)
            self.update_dollars(day)
        self.print_results()
        # self.save_results()

    def print_results(self):
        print 30 * '-'
        print 'Performance for uniform constant rebalancing portfolio:'
        print 30 * '-'
        print 'Total dollar value of assets over time:'
        print self.dollars_history
        print 'Sharpe ratio: TODO'
        #print util.empirical_sharpe_ratio()
        #plt.plot(self.dollars_hist)
        #plt.show()

    def save_results(self):
        output_fname = 'results/const_rebalancing_dollars_over_time.txt'
        print 'Saving dollar value to file: ', output_fname
        output_file = open(output_fname, 'w')
        output_file.write('\t'.join(map(str, self.dollars_hist)))



