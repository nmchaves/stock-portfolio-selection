"""
    This file implements the uniform constant rebalancing portfolio (UCRP).

    UCRP serves as a very simple baseline.
"""

import util
from portfolio import Portfolio


class UniformConstantRebalancedPortfolio(Portfolio):

    def run(self):
        print 'Running UCRP'
        for day in range(0, self.num_days):
            self.update(day)
        self.print_results()
        # self.save_results()

    def get_new_allocation(self, cur_day):
        cur_day_op = self.data.get_op(relative=False)[cur_day, :]  # opening prices on |cur_day|
        new_b = util.get_uniform_allocation(self.num_stocks, cur_day_op)
        return new_b

    def print_results(self):
        print 30 * '-'
        print 'Performance for uniform constant rebalancing portfolio:'
        print 30 * '-'
        Portfolio.print_results(self)
        print self.b_history[-1]

    def save_results(self):
        output_fname = 'results/const_rebalancing_dollars_over_time.txt'
        print 'Saving dollar value to file: ', output_fname
        output_file = open(output_fname, 'w')
        output_file.write('\t'.join(map(str, self.dollars_history)))

