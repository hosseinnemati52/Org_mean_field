#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 12:27:32 2026

@author: hossein
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import re
import subprocess
import os
import time as timetime
from scipy.optimize import newton
import copy
from scipy.optimize import fsolve
import matplotlib as mpl

mpl.rcParams["axes.unicode_minus"] = True

n_sets = 3

folder_names = []
for i in range(n_sets):
    folder_names.append('set_'+str(i+1))


for folder_c in range(len(folder_names)):
    
    folder_name = folder_names[folder_c]
    data_part = np.loadtxt(folder_name+'/WT_lose.csv', delimiter=',')
    
    data_part[np.isnan(data_part)]=0.0
    
    try:
        data_tot = data_tot + data_part.copy()
    except:
        data_tot = data_part.copy()
WT_lose = data_tot.copy()

data_tot = 0.0*data_tot
for folder_c in range(len(folder_names)):
    
    folder_name = folder_names[folder_c]
    data_part = np.loadtxt(folder_name+'/C_gain.csv', delimiter=',')
    
    data_part[np.isnan(data_part)]=0.0
    
    try:
        data_tot = data_tot + data_part.copy()
    except:
        data_tot = data_part.copy()
C_gain = data_tot.copy()


s_list = np.loadtxt("set_1/s_list.csv", delimiter=',')
v_list = np.loadtxt("set_1/v_list.csv", delimiter=',')
s_min = np.min(s_list)
s_max = np.max(s_list)
v_min = np.min(v_list)
v_max = np.max(v_list)
V, S = np.meshgrid(v_list, s_list)

contour_levels_w = np.linspace(np.min(WT_lose), np.max(WT_lose), 12)
contour_levels_c = np.linspace(np.min(C_gain), np.max(C_gain), 12)


plt.figure()
plt.pcolormesh(V, S, WT_lose, shading='auto', cmap='viridis')
# plt.title("change in WT")
plt.xlabel(r'$\tilde{V}_S$', fontsize=20)
plt.ylabel(r'$\tilde{S}$', fontsize=20)
plt.xticks(fontsize=15, rotation=45)
plt.yticks(fontsize=15)
cbar = plt.colorbar()
cbar.set_label(label=r'$\log(\overline{W}_m/\overline{W}_p)$', fontsize=20)
plt.setp(cbar.ax.get_yticklabels(), fontsize=15)
contours = plt.contour(V, S, WT_lose, levels=contour_levels_w, colors='k', linewidths=1, linestyles='solid')
# Optionally add labels to the contour lines
plt.clabel(contours, inline=True, fontsize=13, fmt=lambda x: f"{x:.2g}".replace("-", " −"))
# contours = plt.contour(V, S, phase_quantity_3, levels=contour_levels_g, colors='r', linewidths=1)
# Optionally add labels to the contour lines
# plt.clabel(contours, inline=True, fontsize=8)
plt.scatter([10.7],[19.6], marker=r'$\ast$', color='k', s=100)
# plt.errorbar([10.8],[19.2], xerr=0.1, yerr=0.2, marker='*', color='r')
# plt.xscale('log')
# plt.yscale('log')
plt.tight_layout()
plt.savefig('WT_lose.png', dpi=300)
#plt.show()
# plt.close()


plt.figure()
plt.pcolormesh(V, S, C_gain, shading='auto', cmap='viridis')
# plt.title("change in C")
plt.xlabel(r'$\tilde{V}_S$', fontsize=20)
plt.ylabel(r'$\tilde{S}$', fontsize=20)
plt.xticks(fontsize=15, rotation=45)
plt.yticks(fontsize=15)
cbar = plt.colorbar()
cbar.set_label(label=r'$\log(\overline{C}_m/\overline{C}_p)$', fontsize=20)
plt.setp(cbar.ax.get_yticklabels(), fontsize=15)
contours = plt.contour(V, S, C_gain, levels=contour_levels_c, colors='w', linewidths=1, linestyles='solid')
# Optionally add labels to the contour lines
plt.clabel(contours, inline=True, fontsize=14, fmt='%1.2f')
# contours = plt.contour(V, S, C_gain, levels=contour_levels_c, colors='w', linewidths=1)
# Optionally add labels to the contour lines
# plt.clabel(contours, inline=True, fontsize=11, fmt='%1.2f')
plt.scatter([10.7],[19.6], marker=r'$\ast$', color='k', s=100)
# plt.errorbar([10.8],[19.2], xerr=0.1, yerr=0.2, marker='*', color='r')
# plt.xscale('log')
# plt.yscale('log')
plt.tight_layout()
plt.savefig('C_gain.png', dpi=300)
#plt.show()
# plt.close()


contour_levels_w = np.linspace(np.min(WT_lose), np.max(WT_lose), 12)
contour_levels_c = np.linspace(np.min(C_gain), np.max(C_gain), 12)

plt.figure()
plt.pcolormesh(V, S, 0*WT_lose, shading='auto', cmap='bwr', vmax=1, vmin=-1)
plt.title(r'$\log(\overline{N}_m/\overline{N}_p)$', fontsize=20)
plt.xlabel(r'$\tilde{V}_S$', fontsize=20)
plt.ylabel(r'$\tilde{S}$', fontsize=20)
plt.xticks(fontsize=15, rotation=45)
plt.yticks(fontsize=15)
# plt.colorbar(label="Phase Quantity")
contours = plt.contour(V, S, WT_lose, levels=contour_levels_w, colors='m', linewidths=1.2, linestyles='solid')
plt.clabel(contours, inline=True, fontsize=12,  fmt=lambda x: f"{x:.2g}".replace("-", " −"))
handles_w, labels = contours.legend_elements()
contours = plt.contour(V, S, C_gain, levels=contour_levels_c, colors='g', linewidths=1, linestyles='--')
handles_c, labels = contours.legend_elements()
# Optionally add labels to the contour lines
# contours = plt.contour(V, S, phase_quantity_3, levels=contour_levels_g, colors='r', linewidths=1)
# Optionally add labels to the contour lines
plt.clabel(contours, inline=True, fontsize=12, fmt= '%1.2g')
plt.scatter([10.7],[19.6], marker=r'$\ast$', color='k', s=100)
# plt.errorbar([10.8],[19.2], xerr=0.1, yerr=0.2, marker='*', color='r')
# plt.xscale('log')
# plt.yscale('log')
plt.legend(
    [handles_w[0], handles_c[0]],
    ["WT", "C"],
    fontsize=15,
    loc="upper left",
    bbox_to_anchor=(1.02, 1)
)
plt.tight_layout()
plt.savefig('contours.png', dpi=300)
#plt.show()
# plt.close()