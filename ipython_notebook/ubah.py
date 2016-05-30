import util
from portfolio import Portfolio
from util import get_uniform_allocation


class UniformBuyAndHoldPortfolio(Portfolio):
    """
    Uniform buy and hold portfolio (UBAH).

    UBAH servers as a very simple baseline.
    """

    def run(self):
        print 'Running UCRP'
        for day in range(0, self.num_days):
            self.update(day)
        self.print_results()
        self.save_results()

    def get_new_allocation(self, cur_day):
        if cur_day == 0:
            cur_day_op = self.data.get_op(relative=False)[cur_day, :]  # opening prices on |cur_day|
            return get_uniform_allocation(self.num_stocks, cur_day_op)
        else:
            return self.b

    def print_results(self):
        print 30 * '-'
        print 'Performance for uniform buy and hold strategy:'
        print 30 * '-'
        Portfolio.print_results(self)
        #plt.plot(self.dollars_hist)
        #plt.show()

    def save_results(self):
        output_fname = 'results/uniform_buy_and_hold_dollars_over_time.txt'
        util.save_results(output_fname, self.dollars_history)
