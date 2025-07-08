import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy.ndimage import uniform_filter
import warnings

def sym_X(ax, var, smooth_periods=[1, 5, 10, 15], 
             time_step=5, xlim=[-3, 9], ylim=[-3, 9]):
    """
    Symmetry analysis function for plotting on given axis
    """
    # Load data
    filename = f"/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/mean_{var}.nc"
    data = xr.open_dataset(filename)
    TP = data[var]  # Temperature data (lon, lat, time)
    if var in ['tp','tcrw']:
        TP = TP * 1000
    lat = data['latitude'].values
    lon = data['longitude'].values
    
    # Create coordinate grids and weights
    WT = np.cos(np.radians(lat))  # Weight matrix
    
    YR = np.arange(1940, 2025)
    
    # Calculate anomaly relative to baseline period
    baseline_mean = np.nanmean(TP, axis=0, keepdims=True)
    TPA = TP - baseline_mean
    
    # Loop through smoothing periods
    for sp in smooth_periods:
        # Smooth the data (simple moving average)
        TPS = smooth_data(TPA, sp)
        YRS = YR[sp-1:]  # Adjusted years after smoothing
        
        # Select time points for analysis
        tx = YRS[::time_step]  # Every 'time_step' years
        
        for t in tx:
            # Find index for current time
            inx = np.where(YRS == t)[0]
            if len(inx) == 0:
                continue
                
            M = TPS[:, :, inx[0]]
            
            # Calculate weighted mean
            WM = weighted_mean(M, WT)
            
            # Calculate histogram
            p1, p2 = float(np.min(M).values), float(np.max(M).values)            
            edges = np.linspace(p1, p2, 100)
            ctX, N = weighted_histogram(M, WT, edges)
            
            # Fit parameters using least squares
            m, L1, L2 = f_lsqx(ctX, N)
            
            # Calculate asymmetry analysis
            p = min(abs(ctX[0] - m), abs(ctX[-1] - m))
            a = np.linspace(0, p, 20)
            
            # Interpolate histogram values
            try:
                N1 = np.interp(a + m, ctX, N)
                N2 = np.interp(-a + m, ctX, N)
                
                # Avoid division by zero
                valid_mask = (N1 > 0) & (N2 > 0)
                if np.any(valid_mask):
                    x_vals = a[valid_mask] * (L2 - L1)
                    y_vals = np.log(N1[valid_mask] / N2[valid_mask])
                    
                    # Plot on provided axis
                    ax.plot(x_vals, y_vals, color='gray', alpha=0.7)
            except:
                continue
    
    # Set axis properties
    ax.set_xlabel(r'$\alpha\Delta \beta(\beta_1\beta_2)^{-1}$')
    ax.set_ylabel(r'$\ln(f(\alpha+m)/f(-\alpha+m))$')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

def sym_DX(ax, var, smooth_periods=[5, 10, 15,20], time_step=5,xlim=[-3, 9], ylim=[-3, 9]):
    """
    Symmetry analysis for temperature differences (Delta X) based on FTDA.m
    """
    # Load data
    filename = f"/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/mean_{var}.nc"
    data = xr.open_dataset(filename)
    TP = data[var]  # Temperature data (lon, lat, time)
    if var in ['tp','tcrw']:
        TP = TP * 1000
    lat = data['latitude'].values
    lon = data['longitude'].values
    
    # Create coordinate grids and weights
    WT = np.cos(np.radians(lat))  # Weight matrix
    
    YR = np.arange(1940, 2025)
    
    # Loop through smoothing periods
    for sp in smooth_periods:
        # Smooth the data
        TPS = smooth_data(TP, sp)
        YRS = YR[sp-1:]  # Adjusted years after smoothing
        
        # Set time intervals (DT) similar to MATLAB code
        DT = np.arange(sp-1, len(YRS)-1, time_step)
        
        for dt in DT:
            if dt < len(YRS):
                # Get data for first year and year after interval dt
                X = TPS[:, :, 0]  # First year data
                Y = TPS[:, :, dt] # Data after interval dt
                M = Y - X         # Calculate temperature difference
                
                p1 = max(float(np.nanmin(M)), -8)
                p2 = min(float(np.nanmax(M)), 8)
                
                edges = np.linspace(p1, p2, 100)
                ctX, N = weighted_histogram(M, WT, edges)
                
                # Fit parameters using least squares
                m, L1, L2 = f_lsqx(ctX, N)
                
                # Symmetry analysis
                p = min(abs(ctX[0] - m), abs(ctX[-1] - m))
                a = np.linspace(0, p, 20)
                
                # Interpolate histogram values
                try:
                    N1 = np.interp(a + m, ctX, N)
                    N2 = np.interp(-a + m, ctX, N)
                    
                    # Avoid division by zero
                    valid_mask = (N1 > 0) & (N2 > 0)
                    if np.any(valid_mask):
                        x_vals = a[valid_mask] * (L2 - L1)
                        y_vals = np.log(N1[valid_mask] / N2[valid_mask])
                        
                        # Plot on provided axis (gray color as in MATLAB)
                        ax.plot(x_vals, y_vals, color=[99/255, 99/255, 99/255], alpha=0.7)
                except:
                    continue
    
    # Set axis properties (same as MATLAB)
    ax.set_xlabel(r'$\alpha\Delta \beta(\beta_1\beta_2)^{-1}$')
    ax.set_ylabel(r'$\ln(f(\alpha+m)/f(-\alpha+m))$')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

def smooth_data(data, window_size):

    smoothed = data.copy()
    data_values = data.values
    
    for i in range(data_values.shape[0]):
        for j in range(data_values.shape[1]):
            smoothed.values[i, j, :] = uniform_filter(data_values[i, j, :], size=window_size, mode='nearest')
    
    return smoothed

def weighted_mean(data, weights):
    """Calculate weighted mean"""
    return np.sum(weights * data) / np.sum(weights)

def weighted_histogram(data, weights, edges):
    """Calculate weighted histogram"""
    centers = (edges[:-1] + edges[1:]) / 2
    hist = np.zeros(len(centers))
    # Broadcast weights to match data shape
    weights_2d = np.broadcast_to(weights, data.shape)
    for i in range(len(centers)):
        mask = (data >= edges[i]) & (data < edges[i+1])
        if np.any(mask):
            hist[i] = np.sum(weights_2d[mask])
    hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist
    return centers, hist

def f_lsqx(ctX, N):
    """
    ctX: bin centers (1D array)
    N: histogram values (1D array)
    Returns: m, L1, L2
    """
    # Remove zeros
    mask = N != 0
    ctX = ctX[mask]
    N = N[mask]
    if len(N) == 0:
        return np.nan, np.nan, np.nan

    # Find index of max N
    b = np.argmax(N)

    if b > 0:
        poly1 = np.polyfit(ctX[:b+1], np.log(N[:b+1]), 1)
    else:
        poly1 = [0, 0]

    # Fit right side
    if b < len(N) - 1:
        poly2 = np.polyfit(ctX[b:], np.log(N[b:]), 1)
    else:
        poly2 = [0, 0]

    # Calculate m, L1, L2
    denom = poly1[0] - poly2[0]
    if denom == 0:
        m = np.nan
    else:
        m = (poly2[1] - poly1[1]) / denom
    L1 = -poly2[0]
    L2 = poly1[0]
    return m, L1, L2


# def plot_fluctuation_results(results):
#     """Plot the fluctuation theorem results"""
#     plt.figure(figsize=(8, 6))
    
#     for result in results:
#         plt.plot(result['x'], result['y'], color='gray', alpha=0.7)
    
#     plt.xlabel(r'$\alpha\Delta \beta(\beta_1\beta_2)^{-1}$')
#     plt.ylabel(r'$\ln(f(\alpha+m)/f(-\alpha+m))$')
    
#     plt.tight_layout()
#     return plt.gcf()


