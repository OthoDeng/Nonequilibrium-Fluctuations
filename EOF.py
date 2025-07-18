import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ds = xr.open_dataset('/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/combined_variables.nc')

a = ds['tp'] * 1000
b = ds['viked']
c = ds['t2m']
d = ds['vimdf'] * 1000

var_names = [
'Total Precipitation',
'Vertical integral of divergence of kinetic energy flux',
'2m Temperature',
'Vertical integral of divergence of moisture flux' ]

var_units = ['mm/day', '', 'K', '']
# Cordinates are: lat;lon; year

# 选择纬度范围：南纬30度到北纬30度
latitude_range = range(-30, 31)
a_subset = a.sel(latitude=latitude_range)
b_subset = b.sel(latitude=latitude_range)
c_subset = c.sel(latitude=latitude_range)
d_subset = d.sel(latitude=latitude_range)

# 纬度加权函数
def apply_latitude_weighting(data):
    """应用纬度加权 sqrt(cos(φ))"""
    lat_rad = np.deg2rad(data.latitude)
    weights = np.sqrt(np.cos(lat_rad))
    return data * weights

# 对所有变量应用纬度加权
a_weighted = apply_latitude_weighting(a_subset)
b_weighted = apply_latitude_weighting(b_subset)
c_weighted = apply_latitude_weighting(c_subset)
d_weighted = apply_latitude_weighting(d_subset)

# EOF分解函数
def perform_eof(data, n_components=1):
    """执行EOF分解"""
    # 重塑数据：(time, lat*lon)
    original_shape = data.shape
    data_reshaped = data.values.reshape(original_shape[0], -1)
    
    # 移除NaN值
    valid_mask = ~np.isnan(data_reshaped).any(axis=0)
    data_clean = data_reshaped[:, valid_mask]
    
    # 标准化
    data_mean = np.nanmean(data_clean, axis=0)
    data_std = np.nanstd(data_clean, axis=0)
    data_normalized = (data_clean - data_mean) / data_std
    
    # PCA分解
    pca = PCA(n_components=n_components)
    pc_time_series = pca.fit_transform(data_normalized)
    
    # 重构EOF模态
    eof_pattern = np.full((original_shape[1] * original_shape[2]), np.nan)
    eof_pattern[valid_mask] = pca.components_[0] * data_std
    eof_pattern = eof_pattern.reshape(original_shape[1], original_shape[2])
    
    # 计算解释方差比
    explained_variance = pca.explained_variance_ratio_[0] * 100
    
    return eof_pattern, pc_time_series[:, 0], explained_variance

# 对四个变量进行EOF分解
variables = [a_weighted,b_weighted, c_weighted,d_weighted]


eof_results = []
for var in variables:
    eof_pattern, pc_series, explained_var = perform_eof(var)
    eof_results.append((eof_pattern, pc_series, explained_var))
    # 创建图像
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('EOF Analysis (30°S - 30°N)', fontsize=16, fontweight='bold')

    # 使用GridSpec调整子图比例
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(4, 2, width_ratios=[2.5, 1])  # 第一列宽度为第二列2.5倍
    

    for i, (eof_pattern, pc_series, explained_var) in enumerate(eof_results):
        # EOF模态
        ax1 = fig.add_subplot(gs[i, 0], projection=ccrs.PlateCarree())
        lon = variables[i].longitude.values
        lat = variables[i].latitude.values
        im = ax1.contourf(lon, lat, eof_pattern, levels=20, cmap='RdBu_r', extend='both', transform=ccrs.PlateCarree())
        
        # 添加地理特征
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax1.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax1.add_feature(cfeature.LAND, alpha=0.2, color='gray')
        ax1.add_feature(cfeature.OCEAN, alpha=0.2, color='lightblue')
        
        ax1.set_title(f'{var_names[i]} EOF1',fontsize = 12)  #\n({explained_var:.1f}% variance)', fontsize=12
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.gridlines(draw_labels=True, alpha=0.3)
        cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
        cbar.set_label(var_units[i])

        # 时间序列
        ax2 = fig.add_subplot(gs[i, 1])
        time_coord = variables[0].year.values
        years = [str(t)[:4] for t in time_coord.astype(str)]
        
        # 根据正负模态设置颜色
        colors = ['red' if val > 0 else 'blue' for val in pc_series]
        ax2.bar(range(len(pc_series)), pc_series, color=colors, alpha=0.7)
        
        # ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        ax2.set_title(f'{var_names[i]} PC1 Time Series', fontsize=12)
        ax2.set_xlabel('Time Index')
        ax2.set_ylabel('PC1 Amplitude')
        ax2.grid(True, alpha=0.3)
        if len(years) > 10:
            step = len(years) // 10
            ax2.set_xticks(range(0, len(years), step))
            ax2.set_xticklabels(years[::step], rotation=45)

    plt.tight_layout()
    plt.savefig('/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/EOF_analysis.png', dpi=800, bbox_inches='tight')

# 打印解释方差信息
print("\nEOF Analysis Results:")
print("=" * 50)
for i, (_, _, explained_var) in enumerate(eof_results):
    print(f"{var_names[i]}: {explained_var:.2f}% variance explained")
