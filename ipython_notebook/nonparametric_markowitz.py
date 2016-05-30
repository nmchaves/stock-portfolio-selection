import time

import numpy as np
from cvxpy import *

import util
from portfolio import Portfolio


class NonParametricMarkowitz(Portfolio):
    def __init__(self, market_data, window_len, k, risk_aversion, start_date, portfolio_cap=1.0):
        self.window_len = window_len
        self.k = k
        self.risk_aversion = risk_aversion
        self.start_date = start_date
        self.mu = None 
        self.sigma = None
        self.predicted_return = None
        self.ma_window = 15
        self.cap = portfolio_cap

        assert(self.start_date > self.ma_window,'Your start date needs to be greater than the size of the filter.')
        super(NonParametricMarkowitz,self).__init__(market_data)

    def get_market_window(self, window, day):
        # Compose historical market window, including opening prices
        available = util.get_avail_stocks(self.data.get_op()[day-window+1,:])
        available_inds = np.asarray([i for i in range(497) if available[i] > 0])

        op = self.data.get_op()[day-window+1:day+1,available_inds]
        hi = self.data.get_hi()[day-window+1:day,available_inds]
        lo = self.data.get_lo()[day-window+1:day,available_inds]
        cl = self.data.get_cl()[day-window+1:day,available_inds]
        history = np.concatenate((op, hi, lo, cl)).T
        return history

    def get_new_allocation(self, cur_day):
        """

        :param cur_day:
        :return: A (1 x num_stocks) array of fractions. Each fraction represents the
        amount of the money should be invested in that stock at the end of the day.
        
        If we haven't reached the start date (the day when we have confidence in our mean/covariance info), return 
        a uniform portfolio. Otherwise, perform nonparametric markowitz.
        """

        if cur_day == 0:
            cur_day_op = self.data.get_op(relative=False)[cur_day, :]  # opening prices on |cur_day|
            return util.get_uniform_allocation(self.num_stocks, cur_day_op)
        elif(cur_day < self.start_date):
            available = util.get_avail_stocks(self.data.get_op()[cur_day,:])
            num_available = np.sum(available)
            new_allocation = 1.0/num_available * np.asarray(available)
        else:
            k = self.k
            available = util.get_avail_stocks(self.data.get_op()[cur_day-self.window_len+1,:])
            available_inds = util.get_available_inds(available)
            history = self.get_market_window(self.window_len, cur_day)
            num_available = history.shape[0]
            neighbors = np.zeros((num_available, k))
            history_norms = np.diag(np.dot(history, history.T))

            # Compute k nearest neighbors for each stock
            for i in range(num_available):
                stock = history[i,:]
                neighbors[i,:] = util.k_nearest_neighbors(stock, history, k, history_norms)
        
            # Solve optimization problem 
            l = self.risk_aversion
            neighbors = neighbors.astype(int)
            b = Variable(num_available)
            c = Variable(num_available)
            d = 0
            e = 0

            for i in range(num_available):
                inds = available_inds[neighbors[i,:]]
                m_i = self.mu[inds]   #self.mu[inds]
                S_i = self.sigma[inds,:]
                S_i = S_i[:,inds]
                d += (b[i]*m_i).T*np.ones(k) 
                e += quad_form(b[i]*np.ones(k), S_i)
                    
                constraints = [c>= b, c >= -b, sum_entries(c)==1] #, b <= self.cap, b >= -self.cap] #[b >= 0, np.ones(num_available).T*b == 1]
                objective = Maximize(d-l*e)
                prob = Problem(objective, constraints)
                prob.solve()

                new_allocation = np.zeros(len(available))
                new_allocation[available_inds] = b.value

        return new_allocation

    def update_statistics(self, cur_day):
        '''
        Perform mean and covariance estimation.
        Currently, we assume independence (diagonal sigma) for the first start_date days, then move to the true ML estimate,
        hoping that our covariance does not become singular later on.
        '''
        
        
        # Get the previous day's closing data
        last_close = self.data.get_cl()[cur_day-1,:]
        num_total_stocks = last_close.shape[0]
        num_examples = cur_day-1

        # If the parameters are uninitialized, initialize them
        if(self.mu == None):
            assert(self.sigma == None)
            self.mu = np.zeros(num_total_stocks).astype(float)
            self.sigma = np.zeros((num_total_stocks,num_total_stocks)).astype(float)
         
        if(cur_day > 1):
            N = num_examples-1
            mu_N = self.mu
            self.mu = N/(N+1.0) * self.mu + 1/(N+1.0) * last_close

            # Use a simple moving average filter to estimate today's closing price
            # if(cur_day > self.ma_window):
            #     window = np.arange(cur_day-self.ma_window, cur_day)
            #     self.predicted_return = 1.0/self.ma_window * np.sum(self.data.get_cl()[window,:], axis=0)

            # During initialization, estimate cov as diagonal; oth., estimate full cov.
            if(cur_day >= self.start_date):   
                self.sigma = N/(N+1.0) * (self.sigma+np.outer(mu_N, mu_N)) - np.outer(self.mu, self.mu) + 1/(N+1.0)*np.outer(last_close, last_close)
            else:
                self.sigma = N/(N+1.0) * (self.sigma + np.diag(mu_N**2)) - np.diag(self.mu**2) + 1/(N+1.0)*np.diag(last_close**2)

    def run(self):
        ref_time = time.time()
        for day in range(0, self.num_days):
            #if(day % 100 == 0):
            cur_time = time.time()
            print 'Day' + str(day)
            print('%fs / day.' % (cur_time-ref_time))
            ref_time = cur_time
            self.update_statistics(day)
            self.update(day)

        self.print_results()
        #self.save_results()

    def print_results(self):
        print 30 * '-'
        print 'Performance for Nonparametric Markowitz Portfolio:'
        print 30 * '-'
        Portfolio.print_results(self)

    def save_results(self):
        output_fname = 'results/nonparametric_markowitz_w' + str(self.window_len) + '_k' + str(self.k) + '_e' + self.risk_aversion + '_s' + str(self.start_date) +  '.txt'
        print 'Saving dollar value to file: ', output_fname
        output_file = open(output_fname, 'w')
        output_file.write('\t'.join(map(str, self.dollars_history)))
