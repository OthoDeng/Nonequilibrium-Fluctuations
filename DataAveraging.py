import numpy as np
import xarray as xr

# 定义四个变量
variables = ['tp', 'sst', 'tcwv', 'w']

# 创建字典来存储所有变量的数据
data_vars = {}
coords = None

# 循环加载每个变量的文件
for var in variables:
    filename = f'/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/mean_{var}.nc'
    ds = xr.open_dataset(filename)
    if var == 'w':
        data_vars[var] = ds['vertical_SW']
    else: 
        data_vars[var] = ds[var]
    
    # 使用第一个文件的坐标系统
    if coords is None:
        coords = ds.coords

# 创建新的组合Dataset
combined_ds = xr.Dataset(data_vars, coords=coords)

# 保存为新的.nc文件
output_file = '/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/combined_variables.nc'
combined_ds.to_netcdf(output_file)
print(f"组合数据已保存到 {output_file}")

# 关闭所有打开的数据集
for var in variables:
    filename = f'/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/mean_{var}.nc'
    ds = xr.open_dataset(filename)
    ds.close()


# # 保存为.nc文件
# output1 = f'/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/mean_{var}.nc'
# ds1.to_netcdf(output1)
# print(f"数据已保存到 {output1}")

# # years = np.arange(1940, 2025)
# # ds2 = xr.Dataset(
# #     {
# #         'tp': (['year', 'latitude', 'longitude'], tp.values),
# #     },
# #     coords={
# #         'year': years,
# #         'latitude': AnnualMean['latitude'],
# #         'longitude': AnnualMean['longitude'],
# #     }
# # )


# # # 保存为.nc文件
# # output2 = '/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/mean_tp.nc'
# # ds2.to_netcdf(output2)




