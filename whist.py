import numpy as np

def f_whist(X, W, edges):
    """
    计算加权直方图并归一化为概率密度函数。
    
    参数:
        X: 数据数组
        W: 权重数组 (不需要预先归一化)
        edges: 区间边界数组
        
    返回:
        ctX: 区间中心点
        N: 归一化后的概率密度
    """
    # 输入验证
    if len(X) != len(W):
        raise ValueError("X和W的长度必须相同")
    if len(edges) < 2:
        raise ValueError("edges必须至少包含两个值")
    
    # 计算区间中心点
    ctX = (edges[:-1] + edges[1:]) / 2
    
    # 计算每个区间的宽度
    delta = edges[1:] - edges[:-1]
    
    # 初始化结果数组
    N = np.zeros_like(ctX)
    
    # 向量化实现 - 对每个区间计算加权计数
    for i in range(len(ctX)):
        mask = (edges[i] <= X) & (X < edges[i+1])
        if np.any(mask):
            N[i] = np.sum(W[mask]) / delta[i]
    
    # 归一化为概率密度函数
    integral = np.trapz(N, ctX)
    if integral > 0:
        N = N / integral
    
    return ctX, N