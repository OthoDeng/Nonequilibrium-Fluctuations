import numpy as np

def f_whist(X, W, edges):
    # weighted histocounts
    # X data; W weight (not necessarily normalized)

    
    ctX = (edges[:-1] + edges[1:]) / 2
    
    delta = edges[1:] - edges[:-1]
    
    N = np.empty_like(ctX)
    
    for ii in range(len(ctX)):
        inx = (edges[ii] <= X) & (X< edges[ii+1])
        N[ii] = np.sum(W[inx]) / delta[ii]
    
    # 归一化为概率密度函数
    integral = np.trapz(N, ctX)
    if integral > 0:
        N = N / integral
    else:
        N = np.zeros_like(N)
    
    return ctX, N