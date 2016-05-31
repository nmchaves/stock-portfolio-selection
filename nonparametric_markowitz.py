import numpy as np
from constants import init_dollars
import util
from portfolio import Portfolio

#import matplotlib.pyplot as plt
from cvxpy import *
import pdb

class NonParametricMarkowitz(Portfolio):
    def __init__(self, market_data, market_data_train=None, window_len=10, k=10, risk_aversion=1e-5, start_date=25, start=0, stop=None, 
                 rebal_interval=1, tune_interval=None, tune_length=None,
                 init_b=None, init_dollars=init_dollars, verbose=False, silent=False,
                 past_results_dir=None, new_results_dir=None, repeat_past=False):
        self.portfolio_type = 'NPM'
        self.window_len = window_len
        self.k = k
        self.risk_aversion = risk_aversion
        self.start_date = start_date
        self.mu = None 
        self.sigma = None
        self.startup_time = 10


        if past_results_dir is not None:
            hyperparams_dict = util.load_hyperparams(past_results_dir, ['Window', 'K', 'Risk', 'Start_Date'])
            self.window_len = int(hyperparams_dict['Window'])
            self.k = int(hyperparams_dict['K'])
            self.risk_aversion = hyperparams_dict['Risk']
            self.start_date = hyperparams_dict['Start_Date']
            self.load_state(past_results_dir)

        super(NonParametricMarkowitz,self).__init__(market_data, market_data_train=market_data_train, start=start, stop=stop, 
                    rebal_interval=rebal_interval, tune_interval=tune_interval, tune_length=tune_length, 
                    init_b=None, init_dollars=init_dollars, verbose=False, silent=False, past_results_dir=past_results_dir, 
                    new_results_dir=new_results_dir, repeat_past=repeat_past)
        

    def get_market_window(self, window, day):
        # Compose historical market window, including opening prices
        available = util.get_avail_stocks(self.data.get_op()[day - window + 1, :])
        available_inds = np.asarray([i for i in range(497) if available[i] > 0])

        if(day >= window):
            op = self.data.get_op()[day-window+1:day+1,available_inds]
            hi = self.data.get_hi()[day-window+1:day,available_inds]
            lo = self.data.get_lo()[day-window+1:day,available_inds]
            cl = self.data.get_cl()[day-window+1:day,available_inds]
        elif(self.data_train is not None):
            op = self.data_train.get_op()[day-window+1:, available_inds]
            op = np.concatenate((op,self.data.get_op()[:day+1,available_inds]))

            hi = self.data_train.get_hi()[day-window+1:, available_inds]
            hi = np.concatenate((hi,self.data.get_hi()[:day,available_inds]))

            lo = self.data_train.get_lo()[day-window+1:, available_inds]
            lo = np.concatenate((lo,self.data.get_lo()[:day,available_inds]))

            cl = self.data_train.get_cl()[day-window+1:, available_inds]
            cl = np.concatenate((cl,self.data.get_cl()[:day,available_inds]))

        else:
            raise 'NPM called get_market_window with day<window'
        history = np.concatenate((op, hi, lo, cl)).T
        return history

    def get_new_allocation(self, cur_day, init=False):
        """

        :param cur_day:
        :return: A (1 x num_stocks) array of fractions. Each fraction represents the
        amount of the money should be invested in that stock at the end of the day.
        
        If we haven't reached the start date (the day when we have confidence in our mean/covariance info), return 
        a uniform portfolio. Otherwise, perform nonparametric markowitz.
        """

        self.update_statistics(cur_day)

        if cur_day == 0:
            cur_day_op = self.data.get_op(relative=False)[cur_day, :]  # opening prices on |cur_day|
            return util.get_uniform_allocation(self.num_stocks, cur_day_op)
        elif(cur_day < self.start_date):
            available = util.get_avail_stocks(self.data.get_op()[cur_day, :])
            num_available = np.sum(available)
            new_allocation = 1.0/num_available * np.asarray(available)
        else:

            k = self.k
            available = util.get_avail_stocks(self.data.get_op()[cur_day - self.window_len + 1, :])
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
            for i in range(num_available):
                inds = available_inds[neighbors[i,:]]
                m_i = self.mu[inds] #np.ones(inds.shape[0])
                S_i = self.sigma[inds,:]
                S_i = S_i[:,inds]

                d += -(b[i]*m_i).T*np.ones(k) + l*quad_form(b[i]*np.ones(k), S_i) #+ 0.00005*norm(b-b_last,2) #0.00005
                
            constraints = [c>= b, c >= -b, sum_entries(c)==1]	#, b <= self.cap, b>= - self.cap] #[b >= 0, np.ones(num_available).T*b == 1]
            objective = Minimize(d)
            prob = Problem(objective, constraints)
            prob.solve()

            new_allocation = np.zeros(len(available))
            new_allocation[available_inds] = b.value

            if(cur_day%50 == 0):
                print 'Day %d' % (cur_day)

        return new_allocation

    def update_statistics(self, cur_day):
        '''
        Perform mean and covariance estimation.
        Currently, we assume independence (diagonal sigma) for the first start_date days, then move to the true ML estimate,
        hoping that our covariance does not become singular later on.
        '''
        
        
        # Get the previous day's closing data
        last_close = np.nan_to_num(self.data.get_cl()[cur_day-1,:])
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
	    
            if(cur_day >= self.start_date):   
                self.sigma = N/(N+1.0) * (self.sigma+np.outer(mu_N, mu_N)) - np.outer(self.mu, self.mu) + 1/(N+1.0)*np.outer(last_close, last_close)
            else:
                self.sigma = N/(N+1.0) * (self.sigma + np.diag(mu_N**2)) - np.diag(self.mu**2) + 1/(N+1.0)*np.diag(last_close**2)
    
    def get_hyperparams_dict(self):
        hyperparams = {
            'Window': str(self.window_len),
            'K': str(self.k),
            'Risk': str(self.risk_aversion),
            'Start_Date' : str(self.start_date),
        }
        return hyperparams

    def save_results(self):
        if self.new_results_dir is None:
            return

        print 'Saving ', self.portfolio_type
        save_dir = self.new_results_dir

        util.save_dollars_history(save_dir=save_dir, dollars=self.dollars_op_history, portfolio_type=self.portfolio_type)
        util.save_b_history(save_dir=save_dir, b_history=self.b_history, portfolio_type=self.portfolio_type)
        util.save_hyperparams(save_dir=save_dir, hyperparams_dict=self.get_hyperparams_dict(), portfolio_type=self.portfolio_type)
        self.save_state(save_dir=save_dir)
        return

    def save_state(self, save_dir):
        np.save(save_dir + 'mu.npy', self.mu)
        np.save(save_dir + 'sigma.npy', self.sigma)

    def load_state(self, load_dir):
        self.mu = np.load(load_dir + 'mu.npy')
        self.sigma = np.load(load_dir + 'mu.npy')


    def print_results(self):
        print 30 * '-'
        print 'Performance for Nonparametric Markowitz Portfolio:'
        print 30 * '-'
        Portfolio.print_results(self)

