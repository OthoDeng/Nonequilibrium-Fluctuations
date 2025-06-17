import numpy as np
import xarray as xr

var = input("请输入变量名（如'tcwv'）: ")
# 加载数据
filedirname = '/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/data_stream-moda_stepType-avgua.nc'
data = xr.open_dataset(filedirname)
lat = data['latitude'].values
lon = data['longitude'].values

# 时间转换成年份
AnnualMean = data.groupby('valid_time.year').mean(dim='valid_time')
# cp = AnnualMean['cp']
# tp = AnnualMean['tp']

var1 = AnnualMean[var]


# 计算风速
# WS = np.sqrt(np.array(UP)**2 + np.array(VP)**2)

# 创建新的xarray.Dataset
years = np.arange(1940, 2025)
ds1 = xr.Dataset(
    {
        var: (['year', 'latitude', 'longitude'], var1.values),
    },
    coords={
        'year': years,
        'latitude': AnnualMean['latitude'],
        'longitude': AnnualMean['longitude'],
    }
)


# 保存为.nc文件
output1 = f'/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/mean_{var}.nc'
ds1.to_netcdf(output1)
print(f"数据已保存到 {output1}")

# years = np.arange(1940, 2025)
# ds2 = xr.Dataset(
#     {
#         'tp': (['year', 'latitude', 'longitude'], tp.values),
#     },
#     coords={
#         'year': years,
#         'latitude': AnnualMean['latitude'],
#         'longitude': AnnualMean['longitude'],
#     }
# )


# # 保存为.nc文件
# output2 = '/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/mean_tp.nc'
# ds2.to_netcdf(output2)




