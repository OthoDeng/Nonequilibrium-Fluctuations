import matplotlib.pyplot as plt
from pdfA import pdfA
from mean_var import mean_var
from mean_skewness import mean_skewness

fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=800)  # 12:8 gives each subplot 1:1 aspect ratio
var = input('Enter the variable name (e.g., "tp", "swr"): ')


pdfA(axes[0][0], var, [5])
pdfA(axes[0][1],var, [10])
mean_var(axes[0][2], var)


pdfA(axes[1][0], var, [15])
pdfA(axes[1][1], var, [20],[-3,3])
mean_skewness(axes[1][2], var)

axes[0][0].text(-0.15, 1.05, 'a', transform=axes[0][0].transAxes,
                fontsize=18, fontweight='bold', va='top', ha='left')
axes[0][2].text(-0.15, 1.05, 'b', transform=axes[0][2].transAxes,
                fontsize=18, fontweight='bold', va='top', ha='left')

plt.tight_layout()
plt.savefig('/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/fig2/combined_figure.png')
# plt.show()