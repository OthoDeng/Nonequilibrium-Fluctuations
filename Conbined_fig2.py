import matplotlib.pyplot as plt
from rescaling_X import rescaling_X,rescaling_DX
from symmertry_X import sym_X, sym_DX


fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=800)  
var = input('Enter the variable name (e.g., "tp", "swr"): ')

# Top row: rescaling functions
rescaling_X(axes[0][0], var, xlim=[-15,15])
rescaling_DX(axes[1][0], var, xlim=[-20,20])

# Bottom row: symmetry functions
sym_X(axes[0][1], var,xlim=[-2, 3], ylim=[-3, 4])
sym_DX(axes[1][1], var, xlim=[-2, 6], ylim=[-3, 8])

for i in range(0,2):
    for j in range(0,2):
        letter = chr(ord('a') + i*2  + j)
        axes[i][j].text(-0.15, 1.05, letter, transform=axes[i][j].transAxes,
                fontsize=18, fontweight='bold', va='top', ha='left')

plt.tight_layout()
plt.savefig('/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/fig6/combined_figure.png')
