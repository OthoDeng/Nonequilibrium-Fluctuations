import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from smooth import smooth

def time_series_latband(var, axes=None):
    # Load dataset
    ds = xr.open_dataset(f'/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/mean_{var}.nc')
    # Extract latitude, longitude, and data
    lat = ds['latitude'].values
    lon = ds['longitude'].values
    TP = ds[var].values 

    if var == 'tp' or var == 'tcrw':
        TP = TP * 1000 # Convert to mm

    LON, LAT = np.meshgrid(lon, lat)
    WT = np.cos(np.radians(LAT)) # Shape (lat, lon)
    YR = np.arange(1949, 2026) 

    # Smooth the data
    TP = smooth(TP, 10)

    # Define latitude bands masks
    mask_90S_30S = (LAT >= -90) & (LAT < -30)
    mask_30S_30N = (LAT >= -30) & (LAT <= 30)
    mask_30N_90N = (LAT > 30) & (LAT <= 90)

    # --- Calculate Time Series Statistics for Latitude Bands ---
    mean_90S_30S_ts = []
    mean_30S_30N_ts = []
    mean_30N_90N_ts = []
    var_90S_30S_ts = []
    var_30S_30N_ts = []
    var_30N_90N_ts = []
    skew_90S_30S_ts = []
    skew_30S_30N_ts = []
    skew_30N_90N_ts = []

    num_times = TP.shape[0]

    for t in range(num_times):
        data_t = TP[t, :, :] # Data for current time step, shape (lat, lon)

        # --- Calculations for 90S-30S ---
        wt_band = WT[mask_90S_30S]
        if wt_band.sum() > 0:
            data_t_band = data_t[mask_90S_30S]
            mean_band = np.average(data_t_band, weights=wt_band)
            var_band = np.average((data_t_band - mean_band)**2, weights=wt_band)
            std_band = np.sqrt(var_band)
            skew_band = np.average(((data_t_band - mean_band) / std_band)**3, weights=wt_band) if std_band > 1e-9 else 0.0
        else:
            mean_band, var_band, skew_band = np.nan, np.nan, np.nan
        mean_90S_30S_ts.append(mean_band)
        var_90S_30S_ts.append(var_band)
        skew_90S_30S_ts.append(skew_band)

        # --- Calculations for 30S-30N ---
        wt_band = WT[mask_30S_30N]
        if wt_band.sum() > 0:
            data_t_band = data_t[mask_30S_30N]
            mean_band = np.average(data_t_band, weights=wt_band)
            var_band = np.average((data_t_band - mean_band)**2, weights=wt_band)
            std_band = np.sqrt(var_band)
            skew_band = np.average(((data_t_band - mean_band) / std_band)**3, weights=wt_band) if std_band > 1e-9 else 0.0
        else:
            mean_band, var_band, skew_band = np.nan, np.nan, np.nan
        mean_30S_30N_ts.append(mean_band)
        var_30S_30N_ts.append(var_band)
        skew_30S_30N_ts.append(skew_band)

        # --- Calculations for 30N-90N ---
        wt_band = WT[mask_30N_90N]
        if wt_band.sum() > 0:
            data_t_band = data_t[mask_30N_90N]
            mean_band = np.average(data_t_band, weights=wt_band)
            var_band = np.average((data_t_band - mean_band)**2, weights=wt_band)
            std_band = np.sqrt(var_band)
            skew_band = np.average(((data_t_band - mean_band) / std_band)**3, weights=wt_band) if std_band > 1e-9 else 0.0
        else:
            mean_band, var_band, skew_band = np.nan, np.nan, np.nan
        mean_30N_90N_ts.append(mean_band)
        var_30N_90N_ts.append(var_band)
        skew_30N_90N_ts.append(skew_band)

    # Convert lists to numpy arrays
    mean_90S_30S_ts = np.array(mean_90S_30S_ts)
    mean_30S_30N_ts = np.array(mean_30S_30N_ts)
    mean_30N_90N_ts = np.array(mean_30N_90N_ts)
    var_90S_30S_ts = np.array(var_90S_30S_ts)
    var_30S_30N_ts = np.array(var_30S_30N_ts)
    var_30N_90N_ts = np.array(var_30N_90N_ts)
    skew_90S_30S_ts = np.array(skew_90S_30S_ts)
    skew_30S_30N_ts = np.array(skew_30S_30N_ts)
    skew_30N_90N_ts = np.array(skew_30N_90N_ts)

    # --- Plotting ---
    own_fig = False
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
        own_fig = True

    # Subplot 1: Mean
    axes[0].plot(YR, mean_90S_30S_ts, 'r--', label='90S-30S')
    axes[0].plot(YR, mean_30S_30N_ts, 'b-.', label='30S-30N')
    axes[0].plot(YR, mean_30N_90N_ts, 'k-', label='30N-90N')
    axes[0].set_ylabel(f'$\\mu_A$', fontsize=12)
    axes[0].legend()
    axes[0].grid(True)

    # Subplot 2: Variance
    axes[1].plot(YR, var_90S_30S_ts, 'r--', label='90S-30S')
    axes[1].plot(YR, var_30S_30N_ts, 'b-.', label='30S-30N')
    axes[1].plot(YR, var_30N_90N_ts, 'k-', label='30N-90N')
    axes[1].set_ylabel(f'$\\sigma_A^2$', fontsize=12)
    axes[1].legend()
    axes[1].grid(True)

    # Subplot 3: Skewness
    axes[2].plot(YR, skew_90S_30S_ts, 'r--', label='90S-30S')
    axes[2].plot(YR, skew_30S_30N_ts, 'b-.', label='30S-30N')
    axes[2].plot(YR, skew_30N_90N_ts, 'k-', label='30N-90N')
    axes[2].set_ylabel(f'$\\gamma_A$', fontsize=12)
    axes[2].legend()
    axes[2].grid(True)

    # for ax in axes:
    #     ax.set_xlabel('Year', fontsize=12)

    if own_fig:
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # save_path = f'/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/fig5/TimeSeries_{var}.png'
        # plt.savefig(save_path, dpi=800)
        # print(f"Plot saved to {save_path}")
        # plt.close(fig)
