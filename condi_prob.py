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
import seaborn as sns


# ========== Step 1: Read and prepare data ========== #
def load_data(varname):
    path = f'../ERA5SLP/mean_{varname}.nc'
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



# Get wet/dry masks using tau-based anomalies
wet_mask = precip_anom > 0
dry_mask = precip_anom < 0



# ========== Step 2.5: Prepare data for conditional KDE plots ========== #
# 展平成一维数组，去除NaN
def flatten_valid(xarr):
    return xarr.values.flatten()

tcwv_flat = flatten_valid(tcwv_smooth)
omega_flat = flatten_valid(omega_smooth)
precip_anom_flat = flatten_valid(precip_anom)

# 干/湿掩码
wet_idx = precip_anom_flat > 0
dry_idx = precip_anom_flat < 0

# 干/湿情境下的X/Y
tcwv_wet = tcwv_flat[wet_idx]
omega_wet = omega_flat[wet_idx]
tcwv_dry = tcwv_flat[dry_idx]
omega_dry = omega_flat[dry_idx]

# ========== Step 3: 绘制概率密度等高线图 ========== #
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

# 湿年
sns.kdeplot(
    x=tcwv_wet, y=omega_wet, 
    fill=True, cmap="Blues", 
    ax=axes[0], levels=10, thresh=0.05
)
axes[0].set_title("Wet Years: P(X, Y | Z>0)")
axes[0].set_xlabel("TCWV Anomaly")
axes[0].set_ylabel("Omega Anomaly")

# 干年
sns.kdeplot(
    x=tcwv_dry, y=omega_dry, 
    fill=True, cmap="Oranges", 
    ax=axes[1], levels=10, thresh=0.05
)
axes[1].set_title("Dry Years: P(X, Y | Z<0)")
axes[1].set_xlabel("TCWV Anomaly")
axes[1].set_ylabel("Omega Anomaly")

plt.tight_layout()
plt.savefig('/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/fig7/condi_kde_wet_dry.png', dpi=800)
plt.show()