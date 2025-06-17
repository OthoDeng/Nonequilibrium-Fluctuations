#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter

# 设置matplotlib参数
rcParams['text.usetex'] = True
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['font.size'] = 12

def load_data(var_name):

    file_path = f"/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/mean_{var_name}.nc"
    print(f"Loading data from {file_path}")
    return xr.open_dataset(file_path)

def calculate_weights(lats):

    return np.cos(np.deg2rad(lats))

def calculate_anomalies(data, years, reference_end=30023):

    reference_mask = years <= reference_end
    reference_mean = data.sel(year=reference_mask).mean(dim='year')
    return data - reference_mean

def smooth_data(data, window_size):

    # 假设数据维度为 (lon, lat, time)
    # 只对时间维度进行平滑处理
    smoothed = data.copy()
    data_values = data.values
    
    for i in range(data_values.shape[0]):
        for j in range(data_values.shape[1]):
            smoothed.values[i, j, :] = uniform_filter(data_values[i, j, :], size=window_size, mode='nearest')
    
    return smoothed

def weighted_histogram(data, weights, bins=100):
    """
    计算加权直方图 - 优化版本
    
    参数:
        data: 二维数据数组 (已展平)
        weights: 对应的权重数组 (已展平)
        bins: 直方图区间数量
    
    返回:
        bin_centers: 区间中心
        hist_values: 加权直方图值
    """
    # 计算数据范围
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)
    
  
    edges = np.linspace(min_val, max_val, bins+1)
    bin_centers = (edges[1:] + edges[:-1]) / 2
    
    # NumPy的直方图函数
    valid_mask = ~np.isnan(data)
    hist_values, _ = np.histogram(data[valid_mask], bins=edges, weights=weights[valid_mask])
    
    # 标准化
    if np.sum(hist_values) > 0:  # 增加防止除零错误的检查
        hist_values = hist_values / np.sum(hist_values) * bins / (max_val - min_val)
    
    return bin_centers, hist_values

def fit_distribution(centers, hist):
    """
    使用最小二乘法拟合双指数分布参数 - 优化版本
    
    参数:
        centers: 区间中心
        hist: 直方图值
    
    返回:
        m: 分布中心
        L1: 右侧斜率 
        L2: 左侧斜率
    """
    # 检查输入数据是否有效
    if np.all(np.isnan(hist)) or len(hist) == 0 or np.max(hist) == 0:
        print("警告: 直方图数据无效")
        return 0, -1.0, 1.0
    
    # 首先估计分布中心 (直方图峰值位置)
    max_idx = np.argmax(hist)
    m_init = centers[max_idx]
    
    # 定义指数函数 - 更有效的向量化实现
    def exp_func(x, m, L1, L2):
        result = np.zeros_like(x, dtype=float)
        left_mask = x < m
        right_mask = ~left_mask
        
        if np.any(left_mask):
            result[left_mask] = np.exp((x[left_mask] - m) * L2)
        if np.any(right_mask):
            result[right_mask] = np.exp((x[right_mask] - m) * L1)
        
        return result
    
    # 拟合参数 - 增加边界和最大迭代次数
    try:
        # 增加bounds参数限制合理的拟合范围
        bounds = ([centers[0], -10.0, 0.0], [centers[-1], 0.0, 10.0])
        params, _ = curve_fit(
            exp_func, centers, hist, 
            p0=[m_init, -1.0, 1.0],
            bounds=bounds,
            maxfev=10000  # 增加最大迭代次数
        )
        return params[0], params[1], params[2]
    except (RuntimeError, ValueError) as e:
        print(f"拟合失败: {e}, 使用估计值")
        return m_init, -1.0, 1.0


var_name = input("请输入变量名称 (例如 'slp'): ")

try:

    # 加载数据
    ds = load_data(var_name)
    
    # 提取变量和坐标
    var_data = ds[var_name]
    if var_name == 'tp' or var_name == 'tcrw':
        var_data = var_data * 1000
    lats = ds.latitude.values
    lons = ds.longitude.values
    years = ds.year.values
    
    # 计算权重
    lat_weights = calculate_weights(lats)
    weights = np.ones((len(lons), len(lats)))
    for i in range(len(lons)):
        weights[i, :] = lat_weights
    
    # 计算异常值
    anomalies = calculate_anomalies(var_data, years)
    
    # 设置平滑参数
    smooth_params = [1, 5,10,15]

    colors = ['#66C2A5', '#8DA0CB', '#3288BD', '#A6D9BF']
    
    # 创建图形 - 使用子图替代单个图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 12))
    
    # 对每个平滑参数进行处理
    for sp in smooth_params:
        # 平滑数据
        smoothed = smooth_data(anomalies, sp)
        
        # 选择年份 (考虑平滑窗口)
        valid_years = years[sp-1:]
        selected_years = np.arange(valid_years[0], valid_years[-1]+1, 5)
        
        print(f"Processing smoothing SP: {sp}")
        # 对每个选定年份进行分析
        for year in selected_years:
            if year in valid_years:
                # 提取该年数据
                year_data = smoothed.sel(year=year).values
                
                # 将数据和权重展平为一维数组
                flat_data = year_data.flatten()
                flat_weights = weights.flatten()
                
                # 计算加权直方图
                centers, hist = weighted_histogram(flat_data, flat_weights)
                
                # 拟合分布参数
                m, L1, L2 = fit_distribution(centers, hist)
                
                # 计算用于绘图的值
                y = np.zeros_like(centers)
                left_mask = centers < m
                right_mask = centers >= m
                
                y[left_mask] = (centers[left_mask] - m) * L2
                y[right_mask] = - (centers[right_mask] - m) * L1
                
                # 半对数图
                ax1.semilogy(y, hist, color=colors[smooth_params.index(sp)], label=f"SP={sp}")
    
    # 添加标签
    ax1.set_ylim(bottom=10**-5)
    ax1.set_xlim(left=-50, right=50)
    ax1.set_xlabel(r'$\overline{A}$')
    ax1.set_ylabel(r'$f(\overline{A})$')
    ax1.text(0.07, 1.02, 'a', transform=ax1.transAxes, fontweight='bold')
    

    # 第二个子图：Delta A的分布
    print("Processing Delta A distributions...")
    for sp in smooth_params:
        # 平滑数据
        smoothed = smooth_data(anomalies, sp)
        
        # 选择年份 (考虑平滑窗口)
        valid_years = years[sp-1:]
        first_year = valid_years[0]
        first_year_data = smoothed.sel(year=first_year).values
        
        selected_years = np.arange(valid_years[0], valid_years[-1]+1, 5)
                
        print(f"Processing smoothing SP: {sp}")

        # 对每个选定年份进行分析
        for year in selected_years:
            if year in valid_years and year > first_year:  # 排除第一年自身
                # 提取该年数据
                year_data = smoothed.sel(year=year).values
                
                # 计算Delta A (当前年份减去第一年)
                delta_data = year_data - first_year_data
                
                # 将数据和权重展平为一维数组
                flat_data = delta_data.flatten()
                flat_weights = weights.flatten()
                
                # 计算加权直方图
                centers, hist = weighted_histogram(flat_data, flat_weights)
                
                # 拟合分布参数
                m, L1, L2 = fit_distribution(centers, hist)
                
                # 计算用于绘图的值
                y = np.zeros_like(centers)
                left_mask = centers < m
                right_mask = centers >= m
                
                y[left_mask] = (centers[left_mask] - m) * L2
                y[right_mask] = - (centers[right_mask] - m) * L1
                
                # 半对数图
                ax2.semilogy(y, hist, color=colors[smooth_params.index(sp)], label=f"SP={sp}")
    
    
    # 添加标签
    ax2.set_ylim(bottom=10**-5)
    ax2.set_xlim(left=-50, right=50)
    ax2.set_xlabel(r'$\Delta \overline{A}$')
    ax2.set_ylabel(r'$f(\Delta \overline{A})$')
    ax2.text(0.07, 1.02, 'c', transform=ax2.transAxes, fontweight='bold')
    
    # 保存图形
    plt.tight_layout()
    output_dir = "/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/fig6/"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{var_name}_distribution.png", dpi=800)
    # plt.show()
    
    print(f"分析完成，图形已保存至 {output_dir}")

except Exception as e:
    print(f"发生错误: {e}")