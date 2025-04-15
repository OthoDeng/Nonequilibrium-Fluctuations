import numpy as np
import xarray as xr

# 加载数据
filedirname = '/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/data_1.nc'
data = xr.open_dataset(filedirname)
lat = data['latitude'].values
lon = data['longitude'].values

# 时间转换成年份
AnnualMean = data.groupby('valid_time.year').mean(dim='valid_time')
TP = AnnualMean['tp']
SWR = AnnualMean['avg_snswrf']
LWR = AnnualMean['avg_snlwrf']


# 计算风速
# WS = np.sqrt(np.array(UP)**2 + np.array(VP)**2)

# 创建新的xarray.Dataset
years = np.arange(1940, 2025)
ds1 = xr.Dataset(
    {
        'swr': (['year', 'latitude', 'longitude'], SWR.values),
    },
    coords={
        'year': years,
        'latitude': AnnualMean['latitude'],
        'longitude': AnnualMean['longitude'],
    }
)


# 保存为.nc文件
output1 = '/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/mean_swr.nc'
ds1.to_netcdf(output1)


# print(f"{WP}数据已保存至 {output1}")


ds2 = xr.Dataset(
    {
        'tp': (['year','latitude', 'longitude'],TP.values),
    },
    coords={
        'year': years,
        'latitude': AnnualMean['latitude'],
        'longitude': AnnualMean['longitude'],
    }
)

output2 = '/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/mean_tp.nc'
ds2.to_netcdf(output2)

print(f"数据已保存至 {output2}")

ds1 = xr.Dataset(
    {
        'lwr': (['year', 'latitude', 'longitude'], LWR.values),
    },
    coords={
        'year': years,
        'latitude': AnnualMean['latitude'],
        'longitude': AnnualMean['longitude'],
    }
)


# 保存为.nc文件
output3 = '/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/mean_lwr.nc'
ds1.to_netcdf(output3)