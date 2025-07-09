import numpy as np
import xarray as xr
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats

# 加载数据
ds = xr.open_dataset('/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/combined_variables.nc')
sst = ds['sst'].values  # 海表温度
tcwv = ds['tcwv'].values  # 总柱水汽
tp = ds['tp'] # 总降水
w = ds['w'].values  # 垂直风速

tp_mean = tp.mean(dim='longitude').mean(dim='latitude')  # 平均降水

percentile = 59  
threshold = np.percentile(tp.values, percentile)
# 获取高降水事件对应的时间索引，并提取年份

high_precip_times = tp['year'].values[tp_mean >= threshold]
    
print(f"高降水年份 (前5%): {high_precip_times}")
print(f"降水阈值: {threshold:.6f}")
    
high_precip_data = ds.sel(year=ds.year.isin(high_precip_times))

tp = high_precip_data['tp'].values
sst = high_precip_data['sst'].values
tcwv = high_precip_data['tcwv'].values
w = high_precip_data['w'].values

# 1. 线性关系分析
def analyze_linear_relationship(X, y, variable_names):
    """提取变量间的线性关系并返回统计信息"""
    X = sm.add_constant(X)  # 添加截距项
    model = sm.OLS(y, X).fit()
    
    print(model.summary())
    
    # 返回系数和统计信息
    return {
        'coefficients': model.params,
        'r_squared': model.rsquared,
        'p_values': model.pvalues,
        'conf_intervals': model.conf_int()
    }

# 2. 非线性关系分析 (基于Clausius-Clapeyron)
def analyze_cc_relationship(temperature, water_vapor):
    """分析温度和水汽是否符合CC关系"""
    # Remove NaN values
    mask = ~(np.isnan(temperature) | np.isnan(water_vapor))
    temperature = temperature[mask]
    water_vapor = water_vapor[mask]
    
    # Check if we have enough data points
    if len(temperature) == 0:
        print("No valid data points after removing NaN values")
        return None
    
    # 取对数转换
    log_wv = np.log(water_vapor)
    
    # 线性回归 (log(TCWV) = a + b*T)
    X = temperature.reshape(-1, 1)
    y = log_wv
    
    model = LinearRegression().fit(X, y)
    slope = model.coef_[0]
    r_squared = model.score(X, y)
    
    # CC关系预期斜率约为0.07 (7%/K)
    print(f"观测斜率: {slope:.4f}/K (理论值: ~0.07/K)")
    print(f"R²: {r_squared:.4f}")
    
    # 计算理论预测的水汽
    wv_predicted = np.exp(model.predict(X))
    
    return {
        'slope': slope,
        'r_squared': r_squared,
        'observed': water_vapor,
        'predicted': wv_predicted
    }

# 实例调用
print("SST-TCWV关系分析:")
sst_tcwv_relation = analyze_cc_relationship(sst, tcwv)

print("\nTCWV-TP关系分析 (包含垂直风速):")
# Flatten arrays and remove NaN values
tcwv_flat = tcwv.flatten()
w_flat = w.flatten()
tp_flat = tp.flatten()

# Create mask for valid data points
mask = ~(np.isnan(tcwv_flat) | np.isnan(w_flat) | np.isnan(tp_flat))
tcwv_clean = tcwv_flat[mask]
w_clean = w_flat[mask]
tp_clean = tp_flat[mask]

X_combined = np.column_stack([tcwv_clean, w_clean])
tp_relation = analyze_linear_relationship(X_combined, tp_clean, ['TCWV', 'w'])

# 绘制观测vs理论预测
import matplotlib.pyplot as plt

def plot_comparison(observed, predicted, x_label, y_label):
    """
    Plot a scatter comparison between observed and predicted values.

    Parameters:
        observed (array-like): Observed data values.
        predicted (array-like): Predicted data values.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(observed, predicted, alpha=0.05, s=1,color='tab:blue', edgecolors='none')
    
    # 添加1:1线
    min_val = min(observed.min(), predicted.min())
    max_val = max(observed.max(), predicted.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel(f'Observed {x_label}')
    plt.ylabel(f'Predicted {y_label}')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.title(f'Observed vs Predicted {y_label}')
    plt.grid(False)
    plt.tight_layout()
    
plot_comparison(sst_tcwv_relation['observed'], sst_tcwv_relation['predicted'], 'Observed Water Vapor', 'CC-predicted Water Vapor')
plt.savefig('/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/sst_tcwv_comparison.png',dpi = 800)
plt.close()