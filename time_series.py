import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from smooth import f_smooth

var = input('Enter the variable name (e.g., "tp", "swr"): ')
# Load dataset
ds = xr.open_dataset(f'/Users/ottodeng/Desktop/Fluctuation/ERA5SLp/mean_{var}.nc')




# Extract latitude, longitude, and data
lat = ds['latitude'].values
lon = ds['longitude'].values
TP = ds[var].values 



LON, LAT = np.meshgrid(lon, lat)
WT = np.cos(np.radians(LAT))
YR = np.arange(1949, 2025)

# Smooth the data
TP = f_smooth(TP, 10)


# Define latitude groups
group_90S_30S = (lat >= -90) & (lat < -30)
group_30S_30N = (lat >= -30) & (lat <= 30)
group_30N_90N = (lat > 30) & (lat <= 90)

# Compute weighted mean for each group over time
mean_90S_30S = np.average(TP[:, group_90S_30S, :], axis=(1, 2), weights=WT[group_90S_30S, :])
mean_30S_30N = np.average(TP[:, group_30S_30N, :], axis=(1, 2), weights=WT[group_30S_30N, :])
mean_30N_90N = np.average(TP[:, group_30N_90N, :], axis=(1, 2), weights=WT[group_30N_90N, :])

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(YR, mean_90S_30S, 'r--', label='90S-30S')
plt.plot(YR, mean_30S_30N, 'b-.', label='30S-30N')
plt.plot(YR, mean_30N_90N, 'k-', label='30N-90N')

# Add labels, legend, and title
# plt.xlabel('Year')
plt.ylabel(f'$\\mu_A$', fontsize=20)
plt.legend()
plt.grid()
plt.tight_layout()


plt.savefig(f'/Users/ottodeng/Desktop/Fluctuation/500hpa/fig5/TimeSeriesMean_{var}.png', dpi=600)

var_90S_30S = np.var(TP[:, group_90S_30S, :], axis=(1, 2), ddof=1)
var_30S_30N = np.var(TP[:, group_30S_30N, :], axis=(1, 2), ddof=1)
var_30N_90N = np.var(TP[:, group_30N_90N, :], axis=(1, 2), ddof=1)

# Plotting variance
plt.figure(figsize=(8, 6))
plt.plot(YR, var_90S_30S, 'r--', label='90S-30S')
plt.plot(YR, var_30S_30N, 'b-.', label='30S-30N')
plt.plot(YR, var_30N_90N, 'k-', label='30N-90N')

# Add labels, legend, and title

plt.ylabel(f'$\\sigma_A^2$', fontsize=20)
plt.legend()
plt.grid()
plt.tight_layout()


plt.savefig(f'/Users/ottodeng/Desktop/Fluctuation/500hpa/fig5/TimeSeriesVar_{var}.png', dpi=600)
plt.close() # Close the variance plot figure


def skewness(data):
    n = len(data)
    mean = np.mean(data)
    std_dev = np.std(data)
    if std_dev == 0: # Avoid division by zero
        return 0
    skew = (np.sum((data - mean)**3) / n) / (std_dev**3)
    return skew

# Calculate skewness for each group (flattening the spatial dimensions)
skew_90S_30S = np.array([skewness(TP[i, group_90S_30S, :].flatten()) for i in range(len(YR))])
skew_30S_30N = np.array([skewness(TP[i, group_30S_30N, :].flatten()) for i in range(len(YR))])
skew_30N_90N = np.array([skewness(TP[i, group_30N_90N, :].flatten()) for i in range(len(YR))])

# Plotting skewness
plt.figure(figsize=(8, 6))
plt.plot(YR, skew_90S_30S, 'r--', label='90S-30S')
plt.plot(YR, skew_30S_30N, 'b-.', label='30S-30N')
plt.plot(YR, skew_30N_90N, 'k-', label='30N-90N')

# Add labels, legend, and title
plt.ylabel(f'$\\Delta$ $\\beta_A$', fontsize=20)
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig(f'/Users/ottodeng/Desktop/Fluctuation/500hpa/fig5/TimeSeriesSkew_{var}.png', dpi=600)
