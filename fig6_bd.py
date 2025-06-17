#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter
from scipy.interpolate import interp1d

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
    计算加权直方图 
    
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
    
    参数:
        centers: 区间中心
        hist: 直方图值
    
    返回:
        m: 分布中心
        L1: 右侧斜率 
        L2: 左侧斜率
    """
    # 数据预处理：移除hist为0的点，避免取对数时出错
    valid_idx = hist > 0
    if not np.any(valid_idx):
        print("警告: 直方图数据无效")
        return 0, -1.0, 1.0
    
    centers = centers[valid_idx]
    hist = hist[valid_idx]
    
    # 找出分布峰值位置（最大概率密度点的索引）
    max_idx = np.argmax(hist)
    
    # 分段线性拟合策略：分别对峰值左右两侧进行拟合
    # 对分布左侧数据的对数进行线性拟合
    if max_idx > 0:  # 确保左侧有数据点
        left_slope, left_intercept = np.polyfit(centers[:max_idx+1], np.log(hist[:max_idx+1]), 1)
    else:
        left_slope = 1.0  # 默认值
    
    # 对分布右侧数据的对数进行线性拟合
    if max_idx < len(centers) - 1:  # 确保右侧有数据点
        right_slope, right_intercept = np.polyfit(centers[max_idx:], np.log(hist[max_idx:]), 1)
    else:
        right_slope = -1.0  # 默认值
  
    m_init = centers[max_idx]
    L1_init = right_slope  # 右侧斜率（通常为负）
    L2_init = left_slope   # 左侧斜率（通常为正）
    
    # 定义双指数函数模型
    def exp_func(x, m, L1, L2):
        result = np.zeros_like(x, dtype=float)
        left_mask = x < m
        right_mask = ~left_mask
        
        if np.any(left_mask):
            result[left_mask] = np.exp((x[left_mask] - m) * L2)
        if np.any(right_mask):
            result[right_mask] = np.exp((x[right_mask] - m) * L1)
        
        return result
    
    # 拟合参数
    try:
        # 设置合理的参数边界
        bounds = ([centers[0], -20.0, 0.0], [centers[-1], 0.0, 20.0])
        
        def log_exp_func(x, m, L1, L2):
            return np.log(np.maximum(exp_func(x, m, L1, L2), 1e-10))
        
        params, _ = curve_fit(
            log_exp_func, centers, np.log(hist), 
            p0=[m_init, L1_init, L2_init],
            bounds=bounds,
            maxfev=10000
        )
        return params[0], params[1], params[2]
    except (RuntimeError, ValueError) as e:
        print(f"拟合失败: {e}, 使用估计值")
        # 返回基于线性拟合的估计值
        return m_init, L1_init, L2_init


var_name = input("请输入变量名称 (例如 'slp'): ")

try:

    # 加载数据
    ds = load_data(var_name)
    
    # 提取变量和坐标
    var_data = ds[var_name]
    if var_name == 'tp':
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
    smooth_params = [1,5,10,15]

    colors = ['#66C2A5', '#8DA0CB', '#3288BD', '#A6D9BF']
    
    # 创建图形 - 使用子图替代单个图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
    
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
                
                min_dist = min(abs(centers[0]-m), abs(centers[-1]-m))
                alpha = np.linspace(0, min_dist, 20)

                # 创建插值函数计算对称位置的概率密度
                f_interp = interp1d(centers, hist, bounds_error=False, fill_value=np.nan)
                    
                # 获取中心点两侧对称位置的概率密度
                N1 = f_interp(alpha + m)  # 右侧
                N2 = f_interp(-alpha + m)  # 左侧

                ax1.plot(
                        alpha * (L2 - L1),  # x轴: α*(L2-L1)
                        np.log(N1 / N2),  # y轴: log(N1/N2)
                        color= colors[smooth_params.index(sp)]
                        )

    # 添加标签
 
    # ax1.set_ylim([-3, 9])
    # ax1.set_xlim([-3, 9])
    # ax1.plot([-3, 9], [-3, 9], 'k--', alpha=0.3)
    ax1.set_xlabel(r'$\alpha\Delta \beta(\beta_1\beta_2)^{-1}$')
    ax1.set_ylabel(r'$\ln(f(\alpha+m)/f(-\alpha+m))$')
    ax1.text(0.07, 1.02, 'b', transform=ax1.transAxes, fontweight='bold')
    

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
                min_dist = min(abs(centers[0]-m), abs(centers[-1]-m))
                alpha = np.linspace(0, min_dist, 20)

                # 创建插值函数计算对称位置的概率密度
                f_interp = interp1d(centers, hist, bounds_error=False, fill_value=np.nan)
                    
                # 获取中心点两侧对称位置的概率密度
                N1 = f_interp(alpha + m)  # 右侧
                N2 = f_interp(-alpha + m)  # 左侧

                ax2.plot(
                        alpha * (L2 - L1),  # x轴: α*(L2-L1)
                        np.log(N1 / N2),  # y轴: log(N1/N2)
                        color= colors[smooth_params.index(sp)]
                        )
    # 添加标签
 
    # ax2.set_ylim([-3, 9])
    # ax2.set_xlim([-3, 9])
    # ax2.plot([-3, 9], [-3, 9], 'k--', alpha=0.3)
    ax2.set_xlabel(r'$\alpha\Delta \beta(\beta_1\beta_2)^{-1}$')
    ax2.set_ylabel(r'$\ln(f(\alpha+m)/f(-\alpha+m))$')
    ax2.text(0.07, 1.02, 'd', transform=ax2.transAxes, fontweight='bold')
    
    # 保存图形
    plt.tight_layout()
    output_dir = "/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/fig6/"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{var_name}_symmetry.png", dpi=800)
    # plt.savefig(f"{output_dir}/{var_name}_distribution.pdf", format='pdf')
    # plt.show()
    
    print(f"分析完成，图形已保存至 {output_dir}")

except Exception as e:
    print(f"发生错误: {e}")