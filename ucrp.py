import util
from portfolio import Portfolio


class UniformConstantRebalancedPortfolio(Portfolio):
    """
        This file implements the uniform constant rebalancing portfolio (UCRP).

        UCRP serves as a very simple baseline.
    """
    def __init__(self, market_data, start=0, stop=None, rebal_interval=1, tune_interval=None,
                 init_b=None, verbose=False, silent=False, past_results_dir=None,
                 new_results_dir=None, repeat_past=False):
        self.portfolio_type = 'UCRP'

        super(UniformConstantRebalancedPortfolio, self).__init__(
                                market_data=market_data, start=start, stop=stop, rebal_interval=rebal_interval,
                                init_b=init_b, tune_interval=tune_interval, verbose=verbose, silent=silent,
                                past_results_dir=past_results_dir, new_results_dir=new_results_dir, repeat_past=repeat_past)

    def get_new_allocation(self, cur_day, init=False):
        cur_day_op = self.data.get_op(relative=False)[cur_day, :]  # opening prices on |cur_day|
        new_b = util.get_uniform_allocation(self.num_stocks, cur_day_op)
        return new_b

    def print_results(self):
        if self.verbose:
            print 30 * '-'
            print 'Performance for uniform constant rebalancing portfolio:'
            print 30 * '-'
            Portfolio.print_results(self)

