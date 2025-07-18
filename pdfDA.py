import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from smooth import f_smooth
from whist import f_whist

# Clear environment
plt.close('all')
plt.figure(dpi=800)


var = input('Enter the variable name (e.g., "tp", "swr"): ')

# Load data
data = xr.open_dataset(f'/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/mean_{var}.nc')
lat = data['latitude'].values
lon = data['longitude'].values

# Get variable data
var_data = data[var].values
if var == 'tp' or var == 'tcrw' or var == 'cp':
    var_data = var_data * 1000  # Convert to mm/month

# Create coordinate grids and weights
LON, LAT = np.meshgrid(lon, lat)
WT = np.cos(np.radians(LAT))
YR = np.arange(1940, 2025)

# Calculate anomalies
var_anomaly = var_data - np.mean(var_data, axis=0, keepdims=True)

# Smoothing parameters and plotting
SP = [5, 10, 15,20]  # tau values

for ii, sp in enumerate(SP):
    var_smoothed = f_smooth(var_anomaly, sp)
    YRS = YR[sp-1:]
    
    # Get the first year data after smoothing
    first_year_data = var_smoothed[0, :, :]
    
    # Create time sampling every 5 years
    tx = np.arange(YRS[0], YRS[-1]+1, 5)
    
    plt.subplot(2, 2, ii+1)
    
    # Generate color map
    CL = plt.cm.viridis(np.linspace(0, 1, len(tx)+2))
    CL = np.flipud(CL)
    
    # Plot PDF for each time point (delta anomaly: current year minus first year)
    for jj, year in enumerate(tx):
        year_idx = np.where(YRS == year)[0]
        if len(year_idx) > 0:
            idx = year_idx[0]
            current_year_data = var_smoothed[idx, :, :]
            
            # Calculate delta anomaly (current year minus first year)
            delta_anomaly = current_year_data - first_year_data
            
            # Calculate histogram for delta anomaly
            ctX, N = f_whist(delta_anomaly, WT, np.linspace(np.nanmin(delta_anomaly), np.nanmax(delta_anomaly), 100))
            plt.semilogy(ctX, N, color=CL[jj+2], linewidth=0.5)
    
    plt.title(f'$\\tau={sp}$', fontsize=10)
    plt.xlim([-4, 7])
    plt.ylim(bottom=10**-4)

# Adjust subplot spacing
plt.subplots_adjust(hspace=0.3, wspace=0.3)

plt.savefig(f'/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/fig2/pdfDA_{var.upper()}.png')