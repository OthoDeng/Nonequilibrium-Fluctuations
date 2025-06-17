import numpy as np
from scipy.ndimage import uniform_filter

def f_smooth(TP, span):
    # TP: year, lat, lon
    a, b, c = TP.shape
    TPS = np.zeros((a-span+1, b, c))
    for jj in range(span):
        TPS += TP[jj:a-span+jj+1, :, :]
    TPS /= span
    return TPS


# def f_smooth(data, window_size):

#     # 假设数据维度为 (lon, lat, time)
#     # 只对时间维度进行平滑处理
#     smoothed = data.copy()
#     data_values = data.values()
    
#     for i in range(data_values.shape[0]):
#         for j in range(data_values.shape[1]):
#             smoothed.values[i, j, :] = uniform_filter(data_values[i, j, :], size=window_size, mode='nearest')
    
#     return smoothed
