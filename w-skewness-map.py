import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from smooth import f_smooth


plt.close('all')

ds = xr.open_dataset('/Users/ottodeng/Desktop/Fluctuation/500hpa/annual_mean_w.nc')

lat = ds['latitude'].values
lon = ds['longitude'].values

TP = ds['vertical_SW'].values


LON, LAT = np.meshgrid(lon, lat)
WT = np.cos(np.radians(LAT))
YR = np.arange(1940, 2025)

# 计算偏度
def skewness(data):
    n = len(data)
    mean = np.mean(data)
    std_dev = np.std(data)
    skew = (np.sum((data - mean)**3) / n) / (std_dev**3)
    return skew

# 计算异常值
TPA = TP - np.mean(TP, axis=0, keepdims=True)


# 使用15年窗口平滑
sp = 15
TPS = f_smooth(TPA, sp)

# 创建偏度计算函数，沿时间维度计算每个网格点的偏度
def calc_spatial_skewness(data):
    # data shape: [time, lat, lon]
    # 为每个空间点计算偏度
    skew_map = np.zeros((data.shape[1], data.shape[2]))
    
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            time_series = data[:,i, j]
            if not np.isnan(time_series).all():  # 跳过全NaN的时间序列
                mean = np.nanmean(time_series)
                std_dev = np.nanstd(time_series)
                if std_dev > 0:
                    # 计算偏度
                    skew = np.nansum(((time_series - mean) ** 3)) / (len(time_series) * (std_dev ** 3))
                    skew_map[i, j] = skew
                else:
                    skew_map[i, j] = np.nan
            else:
                skew_map[i, j] = np.nan
                
    return skew_map

# 计算全球偏度分布
skewness_map = calc_spatial_skewness(TPS)

# 创建南极地图
plt.figure(figsize=(10, 10), dpi=600)
# 使用南极投影
ax = plt.axes(projection=ccrs.SouthPolarStereo())

# 设置地图显示范围（南极地区）
ax.set_extent([-180, 180, -90, -55], ccrs.PlateCarree())

# 创建网格数据
LON, LAT = np.meshgrid(lon, lat)

# 绘制偏度分布
cs = ax.pcolormesh(LON, LAT, skewness_map, transform=ccrs.PlateCarree(),
                cmap='RdBu_r', vmin=-2, vmax=2)

# 添加海岸线和国界
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')

# 添加网格线
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# 添加60°S纬线圈
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
ax.plot(np.cos(theta), np.sin(theta), 
        transform=ccrs.SouthPolarStereo(central_longitude=0.0))

# 添加色标
cbar = plt.colorbar(cs, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
cbar.set_label(r'$\beta_A$ (Skewness) for $\tau = 15$ years', fontsize=12)

# 添加标题
plt.title('Antarctica Region Skewness Distribution (15-year window)', fontsize=16)

# 保存图像
plt.savefig('/Users/ottodeng/Desktop/Fluctuation/500hpa/antarctica_skewness_map_w_15.png', dpi=600, bbox_inches='tight')
plt.show()

