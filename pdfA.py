import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from smooth import f_smooth
from whist import f_whist

# 清理环境
plt.close('all')
plt.figure(dpi=600)

# 加载数据
filedirname = '/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/mean_tp.nc'
data = xr.open_dataset(filedirname)
lat = data['latitude'].values
lon = data['longitude'].values


TP = data['tp'].values


LON, LAT = np.meshgrid(lon, lat)
WT = np.cos(np.radians(LAT))
YR = np.arange(1940, 2025)



# 计算异常值
TPA = TP - np.mean(TP, axis=0, keepdims=True)

# 平滑和绘图
SP = [1, 5, 10, 15]  # tau
MN = np.full((len(SP), len(YR)), np.nan)
VAR = np.full((len(SP), len(YR)), np.nan)
SK = np.full((len(SP), len(YR)), np.nan)



for ii, sp in enumerate(SP):

    TPS = f_smooth(TPA, sp)
    YRS = YR[sp-1:]
    tx = np.arange(YRS[0], YRS[-1]+1, 5)
    plt.subplot(3, 2, ii+1)
    CL = plt.cm.grey(np.linspace(0, 1, len(tx)+2))
    CL = np.flipud(CL)
    for jj, year in enumerate(tx):
        year_idx = np.where(YRS==year)[0]
        if len(year_idx)>0:
            idx = year_idx[0]
            M = TPS[idx, :,:]
            M_flat = M.flatten()
            
            p1 = np.min(M_flat[~np.isnan(M_flat)])
            p2 = np.max(M_flat[~np.isnan(M_flat)])
            
            edges = np.linspace(p1, p2, 100)  
            ctX, N = f_whist(M, WT, edges)
            plt.semilogy(ctX, N, color=CL[jj+2],linewidth=0.5)
    
        
    plt.title(f'$\\tau={sp}$', fontsize=12)
    #plt.xlim([-20, 20])
    plt.ylim([10**(-2.2), 10**(4)])


plt.savefig('/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/pdfA_TP.png')