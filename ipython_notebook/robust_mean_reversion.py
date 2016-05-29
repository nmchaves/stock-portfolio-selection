import util

class RobustMeanReversionPortfolio(util.Portfolio):

    def __init__(self, market_data, window, epsilon, tau):
        self.window = window
        self.epsilon = epsilon
        self.tau = tau
        super(RobustMeanReversionPortfolio, self).__init__(market_data)