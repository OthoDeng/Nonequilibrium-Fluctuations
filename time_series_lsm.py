import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from smooth import f_smooth

var = input('Enter the variable name (e.g., "tp", "swr"): ')
# Load dataset
ds = xr.open_dataset(f'/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/mean_{var}.nc')


# Land-sea mask: include only the sea area
lsm = xr.open_dataset(f'/Users/ottodeng/Desktop/Fluctuation/land-sea-mask.nc')

# Ensure lsm grid matches ds grid if necessary (e.g., using interpolation)
# Example: lsm = lsm.interp_like(ds) # Uncomment and adapt if grids differ

# Extract latitude, longitude, and data
lat = ds['latitude'].values
lon = ds['longitude'].values
TP = ds[var].values # Shape (year,lat, lon)

if var == 'tp' or var == 'tcrw':
    TP = TP * 1000 # Convert to mm


# Get land-sea mask data (assuming 'lsm' variable, 1=land, 0=sea)
# Verify the variable name and values in your specific lsm file
mask_data = lsm['lsm'].squeeze().values # Remove potential time dim, ensure shape (lat, lon)
if mask_data.shape != (len(lat), len(lon)):
    raise ValueError(f"Shape mismatch between mask {mask_data.shape} and data grid {(len(lat), len(lon))}")

sea_mask = (mask_data == 0) # Boolean mask for sea points
land_mask = (mask_data == 1) # Boolean mask for land points

LON, LAT = np.meshgrid(lon, lat)
WT = np.cos(np.radians(LAT)) # Latitude weights, shape (lat, lon)
YR = np.arange(1949, 2025) # Assuming time dimension matches this range

# Smooth the data
TP = f_smooth(TP, 10)

# --- Calculate Time Series Statistics for Land and Sea ---
mean_sea_ts = []
mean_land_ts = []
var_sea_ts = []
var_land_ts = []
skew_sea_ts = []
skew_land_ts = []

num_times = TP.shape[0]

for t in range(num_times):
    data_t = TP[t,:, :] # Data for current time step, shape (lat, lon)

    # --- Sea Calculations ---
    wt_sea = WT[sea_mask]
    if wt_sea.sum() > 0:
        data_t_sea = data_t[sea_mask]
        mean_sea = np.average(data_t_sea, weights=wt_sea)
        var_sea = np.average((data_t_sea - mean_sea)**2, weights=wt_sea)
        std_sea = np.sqrt(var_sea)
        skew_sea = np.average(((data_t_sea - mean_sea) / std_sea)**3, weights=wt_sea) if std_sea > 1e-9 else 0.0
    else:
        mean_sea, var_sea, skew_sea = np.nan, np.nan, np.nan

    # --- Land Calculations ---
    wt_land = WT[land_mask]
    if wt_land.sum() > 0:
        data_t_land = data_t[land_mask]
        mean_land = np.average(data_t_land, weights=wt_land)
        var_land = np.average((data_t_land - mean_land)**2, weights=wt_land)
        std_land = np.sqrt(var_land)
        skew_land = np.average(((data_t_land - mean_land) / std_land)**3, weights=wt_land) if std_land > 1e-9 else 0.0
    else:
        mean_land, var_land, skew_land = np.nan, np.nan, np.nan

    mean_sea_ts.append(mean_sea)
    mean_land_ts.append(mean_land)
    var_sea_ts.append(var_sea)
    var_land_ts.append(var_land)
    skew_sea_ts.append(skew_sea)
    skew_land_ts.append(skew_land)

# Convert lists to numpy arrays
mean_sea_ts = np.array(mean_sea_ts)
mean_land_ts = np.array(mean_land_ts)
var_sea_ts = np.array(var_sea_ts)
var_land_ts = np.array(var_land_ts)
skew_sea_ts = np.array(skew_sea_ts)
skew_land_ts = np.array(skew_land_ts)

# --- Plotting ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
fig.suptitle(f'Time Series Statistics for {var.upper()} (Land vs Sea, Weighted)', fontsize=16)

# Subplot 1: Mean
axes[0].plot(YR, mean_sea_ts, label='Sea', color='blue', linestyle='--')
axes[0].plot(YR, mean_land_ts, label='Land', color='green')
axes[0].set_ylabel(f'Mean($\\mu_A$)', fontsize=12)
axes[0].legend()
axes[0].grid(True)
# axes[0].tick_params(axis='labelsize', labelsize=10)


# Subplot 2: Variance
axes[1].plot(YR, var_sea_ts, label='Sea', color='blue', linestyle='--')
axes[1].plot(YR, var_land_ts, label='Land', color='green')
axes[1].set_ylabel(f'Variance ($\\sigma_A^2$)', fontsize=12)
axes[1].legend()
axes[1].grid(True)
# axes[1].tick_params(axis='labelsize', labelsize=10)


# Subplot 3: Skewness
axes[2].plot(YR, skew_sea_ts, label='Sea', color='blue',linestyle='--')
axes[2].plot(YR, skew_land_ts, label='Land', color='green')
axes[2].set_ylabel(f'Skewness ($\\Delta$ $\\beta_A$)', fontsize=12) # Add symbol if desired, e.g., ($\\gamma_A$)
axes[2].legend()
axes[2].grid(True)
# axes[2].tick_params(axis='labelsize', labelsize=10)


# Common X-axis label
for ax in axes:
    ax.set_xlabel('Year', fontsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

# Save the combined figure
save_path = f'/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/fig5/TimeSeries_LSM_{var}.png'
plt.savefig(save_path, dpi=800)
print(f"Plot saved to {save_path}")

plt.close(fig) # Close the figure


# Remove the old separate plotting sections and skewness function definition
# ... (Old code for separate mean plot) ...
# ... (Old code for separate variance plot) ...
# ... (Old skewness function definition) ...
# ... (Old code for separate skewness plot) ...
