import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from smooth import smooth
from matplotlib.ticker import  MaxNLocator


def mean_skewness(ax, var, xlim=[-0.5, 0.5]):
    ds = xr.open_dataset(f'/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/mean_{var}.nc')
    lat = ds['latitude'].values
    lon = ds['longitude'].values
    TP = ds[var].values
    if var in ['tp', 'tcrw']:
        TP = TP * 1000

    LON, LAT = np.meshgrid(lon, lat)
    WT = np.cos(np.radians(LAT))
    YR = np.arange(1940, 2024)

    def skewness(data):
        n = len(data)
        mean = np.mean(data)
        std_dev = np.std(data)
        skew = (np.sum((data - mean)**3) / n) / (std_dev**3)
        return skew

    TPA = TP - np.mean(TP, axis=0, keepdims=True)
    SP = [1, 15] 
    MN = np.full((len(SP), len(YR)), np.nan)
    SK = np.full((len(SP), len(YR)), np.nan)

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
            skew_val = skewness(field.flatten())
            MN[i, yr_idx] = mean_val
            SK[i, yr_idx] = skew_val

    for i, sp in enumerate(SP):
        valid_idx = ~np.isnan(SK[i])
        ax.scatter(MN[i, valid_idx], SK[i, valid_idx], c=colors[i], s=20, alpha=0.7, label=labels[i])
    ax.set_xlabel(f'$\\mu_A$', fontsize=14)
    ax.set_ylabel(f'$\\Delta \\beta_A$', fontsize=14)
    ax.set_xlim(xlim)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

    ax.legend()
    ax.grid(False)


# Legacy code for plotting
# plt.close('all')
# fig, ax = plt.subplots(figsize=(8, 6), dpi=600)

# var = input('Enter the variable name (e.g., "tp", "swr"): ')

# plot_mean_skewness(ax, var)

# plt.tight_layout()

# plt.savefig(f'/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/fig2/skewness_{var}.png')
# # plt.show()

