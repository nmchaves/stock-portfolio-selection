import numpy as np

def run_portfolio(day, test_data):
    """

    :param day:
    :param market_data:
    :return:
    """

    allocation_fpath = 'results/test_allocation.txt'

    if day == 0:
        # Load train and test data.

        # Run our codes and save to output file
        results = [[]]  # num days x num stocks
        np.savetxt(allocation_fpath, results, delimiter='\t')
        return results[0]
    else:
        # Read the current day's results from output file
        results_file = open(allocation_fpath)
        for (i, line) in enumerate(allocation_fpath):
            if i == day:
                vals = line.split('\t')
                cur_alloc = [float(val) for val in vals]
                results_file.close()
                return cur_alloc
