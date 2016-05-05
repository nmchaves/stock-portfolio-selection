import numpy as np
from scipy import io

cost_per_transaction = 0.0005
mat = io.loadmat('../../../Downloads/portfolio.mat')

train_vol = np.array(mat['train_vol'])
print train_vol