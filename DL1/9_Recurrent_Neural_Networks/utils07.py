import numpy as np, numpy.random
import torch

def getdata():
    np.random.seed(2)
    
    T = 20
    L = 1000
    N = 2
    
    x = np.empty((N, L), 'int64')
    x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
    data = np.sin(x / 1.0 / T).astype('float64')
    
    return data

