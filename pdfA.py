import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from smooth import smooth
from whist import f_whist

def pdfA(ax, var, SP):
    # Load data
    data = xr.open_dataset(f'/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/mean_{var}.nc')
    lat = data['latitude'].values
    lon = data['longitude'].values

    var_data = data[var].values
    if var in ['tp', 'tcrw', 'cp']:
        var_data = var_data * 1000  # Convert to mm/month

    LON, LAT = np.meshgrid(lon, lat)
    WT = np.cos(np.radians(LAT))
    YR = np.arange(1940, 2025)

    var_anomaly = var_data - np.nanmean(var_data, axis=0, keepdims=True) 

    # SP = [5, 10, 15, 20]  # tau values

    for ii, sp in enumerate(SP):
        var_smoothed = smooth(var_anomaly, sp)
        YRS = YR[sp-1:]
        tx = np.arange(YRS[0], YRS[-1]+1, 5)
        # Generate color map
        CL = plt.cm.viridis(np.linspace(0, 1, len(tx)+2))
        CL = np.flipud(CL)
        # Plot PDF for each time point
        for jj, year in enumerate(tx):
            year_idx = np.where(YRS == year)[0]
            if len(year_idx) > 0:
                idx = year_idx[0]
                M = var_smoothed[idx, :, :]
                ctX, N = f_whist(M, WT, np.linspace(np.nanmin(M), np.nanmax(M), 100))
                ax.plot(ctX, N, color=CL[jj+2], linewidth=0.5)
        ax.set_title(f'$\\tau={sp}$', fontsize=10)
        
        ax.set_xlim([-4, 4])
        ax.set_yscale('log')
        ax.set_ylim(bottom=10**-3)
    ax.set_ylabel('PDF')


# # Clear environment
# plt.close('all')
# plt.figure(dpi=800)



# fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# pdfA(axs[0, 0], var)
# pdfA(axs[0, 1], var)
# pdfA(axs[1, 0], var)
# pdfA(axs[1, 1], var)

# # Adjust subplot spacing
# plt.subplots_adjust(hspace=0.3, wspace=0.3)

# plt.savefig(f'/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/fig2/pdfA_{var.upper()}.png')