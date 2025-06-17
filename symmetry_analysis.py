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
    """加载NetCDF数据"""
    file_path = f"/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/mean_{var_name}.nc"
    print(f"Loading data from {file_path}")
    return xr.open_dataset(file_path)

def calculate_weights(lats):
    """计算基于纬度的权重系数 (余弦加权)"""
    return np.cos(np.deg2rad(lats))

def smooth_data(data, window_size):
    """时间维度平滑处理"""
    smoothed = data.copy()
    data_values = data.values
    
    for i in range(data_values.shape[0]):
        for j in range(data_values.shape[1]):
            smoothed.values[i, j, :] = uniform_filter(data_values[i, j, :], size=window_size, mode='nearest')
    
    return smoothed

def weighted_histogram(data, weights, bins=100):
    """计算加权直方图"""
    # 计算数据范围，限制在-8到8之间
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)
    min_val = max(min_val, -8)
    max_val = min(max_val, 8)
    
    # 建立均匀的区间
    edges = np.linspace(min_val, max_val, bins+1)
    bin_centers = (edges[1:] + edges[:-1]) / 2
    
    # 计算加权直方图
    valid_mask = ~np.isnan(data)
    hist_values, _ = np.histogram(data[valid_mask], bins=edges, weights=weights[valid_mask])
    
    # 标准化
    if np.sum(hist_values) > 0:  
        hist_values = hist_values / np.sum(hist_values) * bins / (max_val - min_val)
    
    return bin_centers, hist_values

def fit_distribution(centers, hist):
    """使用最小二乘法拟合双指数分布参数"""
    # 检查输入数据是否有效
    if np.all(np.isnan(hist)) or len(hist) == 0 or np.max(hist) == 0:
        print("警告: 直方图数据无效")
        return 0, -1.0, 1.0
    
    # 估计分布中心 (直方图峰值位置)
    max_idx = np.argmax(hist)
    m_init = centers[max_idx]
    
    # 定义指数函数
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
        bounds = ([centers[0], -10.0, 0.0], [centers[-1], 0.0, 10.0])
        params, _ = curve_fit(
            exp_func, centers, hist, 
            p0=[m_init, -1.0, 1.0],
            bounds=bounds,
            maxfev=10000
        )
        return params[0], params[1], params[2]
    except (RuntimeError, ValueError) as e:
        print(f"拟合失败: {e}, 使用估计值")
        return m_init, -1.0, 1.0


    """主函数 - 分析对称性并绘图"""
var_name = input("请输入变量名称 (例如 'msl'): ")
    
try:
        # 加载数据
        ds = load_data(var_name)
        
        # 提取变量和坐标
        var_data = ds[var_name]
        lats = ds.latitude.values
        lons = ds.longitude.values
        years = ds.year.values
        
        # 计算权重
        lat_weights = calculate_weights(lats)
        weights = np.ones((len(lons), len(lats)))
        for i in range(len(lons)):
            weights[i, :] = lat_weights
        
        # 创建图形
        plt.figure(figsize=(8, 6))
        
        # 设置平滑参数
        smooth_params = [1, 5, 10, 15]
        
        # 对每个平滑参数进行处理
        for sp in smooth_params:
            print(f"处理平滑参数: {sp}")
            
            # 平滑数据
            smoothed = smooth_data(var_data, sp)
            
            # 选择年份范围
            valid_years = years[sp-1:]
            time_intervals = np.arange(sp, len(valid_years), 5)
            
            # 获取第一年数据作为基准
            first_year_data = smoothed.isel(year=0).values
            
            # 对每个时间间隔进行分析
            for dt in time_intervals:
                if dt < len(valid_years):
                    # 获取间隔dt年后的数据
                    later_year_data = smoothed.isel(year=dt).values
                    
                    # 计算变化量
                    delta = later_year_data - first_year_data
                    
                    # 将数据和权重展平为一维数组
                    flat_delta = delta.flatten()
                    flat_weights = weights.flatten()
                    
                    # 计算加权直方图
                    centers, hist = weighted_histogram(flat_delta, flat_weights)
                    
                    # 拟合分布参数
                    m, L1, L2 = fit_distribution(centers, hist)
                    
                    # 对称性分析
                    # 找出中心点到两端的最小距离
                    min_dist = min(abs(centers[0]-m), abs(centers[-1]-m))
                    alpha = np.linspace(0, min_dist, 20)
                    
                    # 创建插值函数计算对称位置的概率密度
                    f_interp = interp1d(centers, hist, bounds_error=False, fill_value=np.nan)
                    
                    # 获取中心点两侧对称位置的概率密度
                    N1 = f_interp(alpha + m)  # 右侧
                    N2 = f_interp(-alpha + m)  # 左侧
                    
                   
                  
                    plt.plot(
                        alpha * (L2 - L1),  # x轴: α*(L2-L1)
                        np.log(N1 / N2),  # y轴: log(N1/N2)
                        color='#636363'
                        )
        
        # 添加标签和设置
        plt.xlabel(r'$\alpha\Delta \beta(\beta_1\beta_2)^{-1}$')
        plt.ylabel(r'$\ln(f(\alpha+m)/f(-\alpha+m))$')
        plt.xlim([-3, 9])
        plt.ylim([-3, 9])
        plt.text(0.07, 1.07, 'd', transform=plt.gca().transAxes, fontweight='bold')
        
        # 绘制对角线参考线
        plt.plot([-3, 9], [-3, 9], 'k--', alpha=0.3)
        
        # 保存图形
        output_dir = "/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/fig6/"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/{var_name}_symmetry.png", dpi=800)
        # plt.savefig(f"{output_dir}/{var_name}_symmetry.pdf", format='pdf')
        plt.tight_layout()
        plt.show()
        
        print(f"分析完成，图形已保存至 {output_dir}")
        
except Exception as e:
        print(f"发生错误: {e}")


