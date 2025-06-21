import numpy as np
# from scipy.ndimage import uniform_filter

def smooth(TP, span):
    # TP: year, lat, lon
    a, b, c = TP.shape
    TPS = np.zeros((a-span+1, b, c))
    for jj in range(span):
        TPS += TP[jj:a-span+jj+1, :, :]
    TPS /= span
    return TPS



# def smooth(data, window_size):
#     """
#     Smooths the input data using a simple moving average.

#     Args:
#         data (list): The input data to smooth.
#         window_size (int): The size of the moving average window.

#     Returns:
#         list: The smoothed data.
#     """
#     if window_size <= 0:
#         raise ValueError("Window size must be greater than 0")
    
#     smoothed_data = []
#     for i in range(len(data)):
#         start = max(0, i - window_size + 1)
#         end = i + 1
#         window = data[start:end]
#         smoothed_data.append(sum(window) / len(window))
    
#     return smoothed_data
