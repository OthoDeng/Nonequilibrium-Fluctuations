import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from smooth import smooth


def mean_var(ax, var):
    ds = xr.open_dataset(f'/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/mean_{var}.nc')
    lat = ds['latitude'].values
    lon = ds['longitude'].values
    TP = ds[var].values
    if var in ['tp', 'tcrw']:
        TP = TP * 1000

    LON, LAT = np.meshgrid(lon, lat)
    WT = np.cos(np.radians(LAT))
    YR = np.arange(1940, 2025)

    TPA = TP - np.mean(TP, axis=0, keepdims=True)
    SP = [1, 15] 
    MN = np.full((len(SP), len(YR)), np.nan)
    VAR = np.full((len(SP), len(YR)), np.nan)

    colors = ['blue', 'red']
    labels = [r'$\tau = 1$', r'$\tau = 15$']

    for i, sp in enumerate(SP):
        TPS = smooth(TPA, sp)
        YRS = YR[sp-1:]
        for j, yr in enumerate(YRS):
            yr_idx = np.where(YR == yr)[0][0]
            field = TPS[j]
            weights = WT 
            mean_val = np.mean(field * weights)
            var_val = np.nansum(((field - mean_val)**2) * weights)
            MN[i, yr_idx] = mean_val
            VAR[i, yr_idx] = var_val

    for i, sp in enumerate(SP):
        valid_idx = ~np.isnan(MN[i]) & ~np.isnan(VAR[i])
        ax.scatter(MN[i, valid_idx], VAR[i, valid_idx], c=colors[i], s=20, alpha=0.7, label=labels[i])
    ax.set_xlabel(f'$\\mu_A$', fontsize=14)
    ax.set_ylabel(f'$\\sigma^2_A$', fontsize=14)
    ax.legend()
    ax.grid(False)



# plt.close('all')
# fig, ax = plt.subplots(figsize=(8, 8), dpi=600)

# var = input('Enter the variable name (e.g., "tp", "swr"): ")

# mean_var(ax, var)

# plt.tight_layout()

# plt.savefig(f'/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/fig2/mean_var_{var}.png')
# # plt.show()