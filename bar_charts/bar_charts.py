# -*- coding: utf-8 -*-
"""
Created on Sun May 25 22:56:08 2025

@author: Nemat002
"""

import numpy as np
import matplotlib.pyplot as plt


bar_width=0.28

# G1
categories1 = ['pure WT', 'mixed WT']
values1 = [55, 60]
errors1 = [12, 20]
scatter_y1 = [46, 51]
scatter_err1 = [1, 1]

# S/G2/M
categories2 = ['pure WT', 'mixed WT']
values2 = [20, 10]
errors2 = [2, 2]
scatter_y2 = [34, 19]
scatter_err2 = [1, 1]
# Define custom x-positions to group bars: (group 1 near each other, group 2 farther away)
x2 = np.array([0.3, 1.5])

# G0
categories3 = ['pure WT', 'mixed WT']
values3 = [47, 30]
errors3 = [2,2]
x3 = np.array([0, 1.2])



values4 = [20, 29]
errors4 = [2, 2]
scatter_y4 = [20, 30]
scatter_err4 = [1, 1]

# x-positions for panel 1
x1 = 0.5*np.arange(len(categories1))

# Create subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4), sharey=False, constrained_layout=True)

# Bar chart 1
ax1.bar(x1, values1, yerr=errors1, capsize=5, alpha=0.6, label='exp (FUCCI2)', width=bar_width, color='b')
ax1.errorbar(x1, scatter_y1, yerr=scatter_err1, fmt='o', color='red', capsize=10, label='model', markersize=10)
ax1.plot(x1, scatter_y1, color='red', linewidth=1, linestyle='--')
ax1.set_xticks(x1)
ax1.set_xticklabels(categories1, fontsize=15)
ax1.set_ylabel('G1 percentage', fontsize=15)
ax1.tick_params(axis='y', labelsize=15)
ax1.legend(fontsize=15)



# Bar chart 2 (with custom spacing)
ax2.bar(x2, values2, yerr=errors2, capsize=5, alpha=0.6, label='exp (FUCCI2)', width=bar_width, color ='b')
# ax2.errorbar(x2, scatter_y2, yerr=scatter_err2, fmt='o', color='blue', capsize=5, label='Scatter')
# ax2.set_xticks(x2)
# ax2.set_xticklabels(categories2, fontsize=15)
# ax2.set_ylabel('S/G2/M percentage', fontsize=15)
# ax2.tick_params(axis='y', labelsize=15)
# ax2.legend(fontsize=15)
# ax2.legend(fontsize=15)

# Bar chart 2 (with custom spacing)
ax2.bar(x3, values3, yerr=errors3, capsize=5, alpha=0.6, label='exp (EdU/pH3)', width=bar_width, color = 'r')
ax2.errorbar(0.5*(x2+x3), scatter_y2, yerr=scatter_err2, fmt='o', color='r', capsize=10, label='model', markersize=10)
ax2.plot(0.5*(x2+x3), scatter_y2, color='r', linewidth=1, linestyle='--')
ax2.set_xticks(0.5*(x2+x3))
ax2.set_xticklabels(categories2, fontsize=15)
ax2.set_ylabel('S/G2/M percentage', fontsize=15)
ax2.tick_params(axis='y', labelsize=15)
ax2.legend(fontsize=15)

# Bar chart 2 (with custom spacing)
ax3.bar(x1, values4, yerr=errors4, capsize=5, alpha=0.6, label='exp (FUCCI2)', width=bar_width, color = 'b')
ax3.errorbar(x1, scatter_y4, yerr=scatter_err4, fmt='o', color='r', capsize=10, label='model', markersize=10, alpha=0.6)
ax3.plot(x1, scatter_y4, color='r', linewidth=1, linestyle='--')
ax3.set_xticks(x1)
ax3.set_xticklabels(categories3, fontsize=15)
ax3.set_ylabel('G0 percentage', fontsize=15)
ax3.tick_params(axis='y', labelsize=15)
ax3.legend(fontsize=15)

# Layout and show
# plt.tight_layout()


handles, labels = ax2.get_legend_handles_labels()

# Add a single legend for the whole figure
fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=15, frameon=True, bbox_to_anchor=(0.5, 1.02))

# Optionally remove individual legends
ax1.legend().remove()
ax2.legend().remove()
ax3.legend().remove()

# plt.tight_layout()
plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig('phase_percentages.PNG', dpi=400)
# plt.show()