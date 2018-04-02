import numpy as np

def sparse_dim_1(ndaray, sp):
    length = len(ndaray)
    n = length / sp
    ndaray_sparse = [ndaray[i * sp] for i in range(n)]
    if length % sp !=0:
        ndaray_sparse.append(ndaray[-1])
    return np.array(ndaray_sparse)

def sparse_dim_n(ndaray, sp):

    length = len(ndaray)
    n = length / sp
    ndaray_sparse = [ndaray[i * sp] for i in range(n)]
    if length % sp !=0:
        ndaray_sparse.append(ndaray[-1])
    return np.array(ndaray_sparse)
