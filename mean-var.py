import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from smooth import f_smooth


plt.close('all')
plt.figure(figsize=(8, 8), dpi=600)

var = input('Enter the variable name (e.g., "tp", "swr"): ')

ds = xr.open_dataset(f'/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/mean_{var}.nc')

lat = ds['latitude'].values
lon = ds['longitude'].values

TP = ds[var].values
if var == 'tp' or var == 'tcrw':
    TP = TP * 1000  # Convert to mm/month


LON, LAT = np.meshgrid(lon, lat)
WT = np.cos(np.radians(LAT))
YR = np.arange(1940, 2025)



# 计算异常值
TPA = TP - np.mean(TP, axis=0, keepdims=True)

SP = [1, 15] 
MN = np.full((len(SP), len(YR)), np.nan)
VAR = np.full((len(SP), len(YR)), np.nan)
SK = np.full((len(SP), len(YR)), np.nan)

# 计算每个tau值下的平均值和方差
colors = ['blue', 'red']
labels = [r'$\tau = 1$', r'$\tau = 15$']

for i, sp in enumerate(SP):
    TPS = f_smooth(TPA, sp)
    YRS = YR[sp-1:]
    
    # 计算每年的加权全球平均值和方差
    for j, yr in enumerate(YRS):
        yr_idx = np.where(YR == yr)[0][0]
        field = TPS[j]  # 当前年份的平滑场
        
        # 使用余弦权重计算全球平均值
        weights = WT 
        mean_val = np.mean(field * weights)
        var_val = np.nansum(((field - mean_val)**2) * weights)
        
        MN[i, yr_idx] = mean_val
        VAR[i, yr_idx] = var_val

# 绘制平均值-方差散点图
plt.figure(figsize=(8, 6), dpi=600)
for i, sp in enumerate(SP):
    valid_idx = ~np.isnan(MN[i]) & ~np.isnan(VAR[i])
    plt.scatter(MN[i, valid_idx], VAR[i, valid_idx], c=colors[i], s=20, alpha=0.7, label=labels[i])
    

# plt.xlim([-0.01, 0.01])
# plt.ylim(top = 2.7)
plt.xlabel(f'$\\mu_A$', fontsize=14)
plt.ylabel(f'$\\sigma^2_A$', fontsize=14)
# plt.title('Total Precipitation', fontsize=16)
plt.legend()
plt.grid(False)
# plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig(f'/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/fig2/mean_var_{var}.png')
# plt.show()