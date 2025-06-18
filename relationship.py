import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.stats import rankdata, norm, kendalltau
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from smooth import smooth  # Assuming smooth is defined in a separate module
from tau_anomalies import tau__anomalies
import cmocean.cm as cmo
from copulas.multivariate import GaussianMultivariate


# ========== Step 1: Read and prepare data ========== #
def load_data(varname):
    path = f'ERA5SLP/mean_{varname}.nc'
    ds = xr.open_dataset(path)
    if varname == 'w':
        var_data = ds['vertical_SW'] 
    else:
        var_data = ds[varname]
    # Ensure dimensions are in (lon, lat, time) order
    if var_data.dims != ('longitude', 'latitude', 'year'):
        var_data = var_data.transpose('longitude', 'latitude', 'year')
    # Rename dimensions for consistency
    return var_data


# Load and align variables
tcwv = load_data('tcwv')        # Total Column Water Vapor
omega = load_data('w')          # Vertical Wind 
precip = load_data('tp')        # Total Precipitation

tau = 5  # Define tau threshold (adjust as needed)

# Smooth the data
tcwv_smooth = tcwv.rolling(year=tau, center=True).mean()
omega_smooth = omega.rolling(year=tau, center=True).mean()
precip_smooth = precip.rolling(year=tau, center=True).mean()

print("Data loaded and smoothed successfully.")

# Align dimensions & time
tcwv_smooth, omega_smooth, precip_smooth = xr.align(tcwv_smooth, omega_smooth, precip_smooth)

# ========== Step 2: Calculate precipitation anomalies using tau ========== #
# Calculate standard anomalies
precip_anom = precip_smooth - precip_smooth.mean(dim='year')

# Calculate tau-based anomalies

tau_anomalies = tau__anomalies(precip_smooth, tau=tau)

# Get wet/dry masks using tau-based anomalies
wet_mask = tau_anomalies > 0
dry_mask = tau_anomalies < 0

# ========== Step 3: Build function, fit Gaussian Copula ========== #
def empirical_cdf(x):
    """Convert data to [0, 1] interval (empirical CDF)"""
    ranks = rankdata(x)
    return ranks / (len(x) + 1)
def fit_gaussian_copula(x, y):
    """Fit Gaussian Copula and return correlation coefficient rho"""
    u = empirical_cdf(x)
    v = empirical_cdf(y)
    data_uv = np.column_stack((u, v))
    cop = GaussianMultivariate()
    cop.fit(data_uv)
    return cop.covariance[0, 1]  # Return rho

# ========== Step 4: Fit copula for each grid point ========== #
lon = tcwv_smooth.longitude.values
lat = tcwv_smooth.latitude.values
rho_dry = np.full((len(lon), len(lat)), np.nan)
rho_wet = np.full((len(lon), len(lat)), np.nan)

for i in tqdm(range(len(lon))):
    for j in range(len(lat)):
        # Extract time series for each variable at this grid point
        x = tcwv_smooth.isel(longitude=i, latitude=j).values
        y = omega_smooth.isel(longitude=i, latitude=j).values
        p = tau_anomalies.isel(longitude=i, latitude=j).values

        if np.any(np.isnan(x)) or np.any(np.isnan(y)) or np.any(np.isnan(p)):
            continue

        # Get indices for wet and dry years
        wet_idx = wet_mask.isel(longitude=i, latitude=j).values
        dry_idx = dry_mask.isel(longitude=i, latitude=j).values

        # Fit copula
        try:
            if dry_idx.sum() > 10:
                rho_dry[i, j] = fit_gaussian_copula(x[dry_idx], y[dry_idx])
            if wet_idx.sum() > 10:
                rho_wet[i, j] = fit_gaussian_copula(x[wet_idx], y[wet_idx])
        except:
            continue

# Difference map
rho_diff = rho_wet - rho_dry

# ========== Step 5: Plotting with publication-quality colors ========== #
def plot_map(data, title, vmin=-1, vmax=1, cmap=cmo.balance):
    """Plot a single map with modern journal colors"""
    fig = plt.figure(figsize=(10, 6), dpi=150)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.8, color='black')
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    
    im = ax.pcolormesh(lon, lat, data.T, cmap=cmap, vmin=vmin, vmax=vmax, 
                      transform=ccrs.PlateCarree())
    
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, 
                      shrink=0.8, extend='both')
    cbar.ax.tick_params(labelsize=9)
    
    plt.title(title, fontsize=12, fontweight='bold')
    return fig

# Create publication-quality subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6), 
                       subplot_kw={'projection': ccrs.PlateCarree()})

titles = ['Dry Years: Copula Rho', 'Wet Years: Copula Rho', 'Wet - Dry Difference']
data_list = [rho_dry, rho_wet, rho_diff]
cmaps = [cmo.ice, cmo.amp, cmo.balance]
vranges = [(-0.8, 0.8), (-0.8, 0.8), (-0.5, 0.5)]

for i, (ax, data, title, cmap, vrange) in enumerate(zip(axs, data_list, titles, cmaps, vranges)):
    ax.coastlines(linewidth=0.8, color='black')
    ax.gridlines(draw_labels=True if i == 0 else False, 
                linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    
    im = ax.pcolormesh(lon, lat, data.T, transform=ccrs.PlateCarree(),
                      cmap=cmap, vmin=vrange[0], vmax=vrange[1])
    
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                      pad=0.05, shrink=0.8, extend='both')
    cbar.ax.tick_params(labelsize=9)
    ax.set_title(title, fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/fig7/relationship_analysis.png', 
           bbox_inches='tight', dpi=800)
plt.show()