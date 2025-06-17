import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression


# --- Load Data ---
# Ensure paths are correct
ds1 = xr.open_dataset('/Users/ottodeng/Desktop/Fluctuation/annual_mean_sst.nc')
sst_raw = ds1['sst']

# Calculate SST anomaly
sst_mean = sst_raw.mean(dim='year')
sst = sst_raw - sst_mean
sst.name = 'sst_anomaly' # Optional: rename for clarity

ds2 = xr.open_dataset('/Users/ottodeng/Desktop/Fluctuation/1000hpa/annual_mean_w.nc')
w_raw = ds2['vertical_SW']

# Calculate W anomaly
w_mean = w_raw.mean(dim='year')
w = w_raw - w_mean
w.name = 'vertical_SW_anomaly' # Optional: rename for clarity


# --- Align Data (if necessary) ---
# Example: Align w to sst's grid and time

w_aligned = w.interp_like(sst)


# --- Calculate Mutual Information Spatially ---


# Define a function to compute MI for a single grid point's time series
def calculate_mi(x, y):
    # Check for NaNs in inputs for the current grid point's time series
    if np.isnan(x).any() or np.isnan(y).any():
        return np.nan  # Return NaN if any input contains NaN

    # Reshape x to be 2D (n_samples, n_features=1)
    X = x.reshape(-1, 1)
    # y should already be 1D due to input_core_dims

    # Ensure y is 1D (optional, but good practice)
    if y.ndim != 1:
        # This case might indicate an issue with how apply_ufunc passes data
        # print(f"Warning: y is not 1D: {y.shape}") # Uncomment for debugging
        return np.nan # Or handle appropriately

    # mutual_info_regression returns an array, get the first element
    try:
        mi = mutual_info_regression(X, y, discrete_features=False, random_state=0)[0]
        return mi
    except ValueError as e:
        # Catch potential errors during MI calculation itself (e.g., insufficient samples after NaN handling if you chose to drop NaNs)
        print(f"Error during mutual_info_regression: {e}. Shapes: X={X.shape}, y={y.shape}")
        return np.nan

# Use xarray's apply_ufunc to apply the function across grid points
# Input core dimensions are ['time'], output is scalar, vectorize=True speeds it up
mi_map = xr.apply_ufunc(
    calculate_mi,
    w_aligned,  # Feature X
    sst,        # Target y
    input_core_dims=[['year'], ['year']],
    output_core_dims=[[]], # Explicitly define scalar output
    exclude_dims=set(('year',)), # Keep lat/lon dims
    vectorize=True,       # Use numpy vectorization
    dask='allowed',       # Allow dask arrays but rely on vectorize
    output_dtypes=[float] # Specify output type
).rename('mutual_information')

# --- Visualize the Mutual Information Map ---
plt.figure(figsize=(10, 6),dpi=800)
mi_map.plot(cmap='plasma') # Or choose another colormap like 'plasma', 'inferno'
plt.title('Mutual Information between SST and Shortwave Radiation')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# You can print the resulting MI map data
print("Mutual Information Map:")
print(mi_map)

