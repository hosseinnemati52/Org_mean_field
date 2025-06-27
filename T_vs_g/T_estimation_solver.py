# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:49:31 2025

@author: Nemat002
"""

import numpy as np
from scipy.optimize import newton
import matplotlib.pyplot as plt


# Example: Solve f(x) = x^3 + x - 1 = 0
def f(x):
    return frac*x**(g/2) + x - 2

# Optional: derivative
def fprime(x):
    return (g/2) * frac*x**((g/2)-1) + 1.0


frac_list = [0.1, 0.2, 0.3]
# frac_list = [0.2]
g_list = np.linspace(0.01, 0.99, 100)
# g_list = np.linspace(0.45, 0.55, 100)

T_mat = np.zeros((len(frac_list),len(g_list)))


for i in range(len(frac_list)):
    for j in range(len(g_list)):
        # Initial guess
        x0 = 1.0
        
        frac = frac_list[i]
        g = g_list[j]
        beta = 0.0287
        
        root = newton(f, x0, fprime)
        
        T = (1/beta) * np.log(root)
        
        T_mat[i,j] = T
        
        print("T found:", T)
        

np.savetxt('T_estim.csv', fmt='%1.5f', X=T_mat, delimiter=',')


plt.figure()
plt.plot(g_list, T_mat[0,:], label=r'$f_D$ = 0.1')
plt.plot(g_list, T_mat[1,:], label=r'$f_D$ = 0.2')
plt.plot(g_list, T_mat[2,:], label=r'$f_D$ = 0.3')
# plt.fill_between(np.linspace(0.3, 0.5, 100), y1=15, y2=23, color='y', alpha=0.2)
# plt.plot(g_list, T_mat[3,:], label='f = 0.99')
# plt.grid()
plt.ylim((15,23))
plt.legend(fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.xlabel(r'$g$', fontsize = 15)
plt.ylabel(r'$T$'+'(h)', fontsize = 15)
plt.tight_layout()
plt.savefig('T_vs_g.PNG', dpi=300)
