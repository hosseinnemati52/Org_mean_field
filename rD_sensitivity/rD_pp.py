#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 17:01:19 2026

@author: hossein
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# Example input (replace with your own data)
r_D_list = np.loadtxt('r_D_list.csv', delimiter=',')
r_mu_list = np.loadtxt('r_mu_list.csv', delimiter=',')

# y = r_D_list
# x = r_mu_list
# X, Y = np.meshgrid(r_mu_list, r_D_list)

w_p_norm_final = np.loadtxt('w_p_norm_final.csv', delimiter=',')
c_p_norm_final = np.loadtxt('c_p_norm_final.csv', delimiter=',')
w_m_norm_final = np.loadtxt('w_m_norm_final.csv', delimiter=',')
c_m_norm_final = np.loadtxt('c_m_norm_final.csv', delimiter=',')
wd_p_perc_final = np.loadtxt('wd_p_perc_final.csv', delimiter=',')
wd_m_perc_final = np.loadtxt('wd_m_perc_final.csv', delimiter=',')


# W^* p
plt.figure()
Z = w_p_norm_final
plt.pcolormesh(r_D_list, r_mu_list, Z.T)
plt.colorbar()
plt.scatter([0.43], [3.92], marker=r'$\ast$', color='r', s=200)
plt.scatter([np.min(r_D_list)], [np.min(r_mu_list)], marker=r'$\ast$', color='w', s=0.0001)
plt.scatter([np.max(r_D_list)], [np.max(r_mu_list)], marker=r'$\ast$', color='w', s=0.0001)
plt.xlabel(r'$r_D$', fontsize=20)
plt.xticks(fontsize=12)
plt.xscale("log", base=2)
plt.ylabel(r'$r_\mu=\mu_C/\mu_W$', fontsize=20)
plt.yscale("log", base=2)
plt.yticks(fontsize=20)
plt.title(r'$W^{\ast}_p$', fontsize=20)
plt.gca().xaxis.set_major_locator(ticker.LogLocator(base=2))
plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
plt.tight_layout()
plt.savefig("w_p.png", dpi=300)
plt.show()
# W^* p


# W^* m
plt.figure()
Z = w_m_norm_final
plt.pcolormesh(r_D_list, r_mu_list, Z.T)
plt.colorbar()
plt.scatter([0.43], [3.92], marker=r'$\ast$', color='r', s=200)
plt.scatter([np.min(r_D_list)], [np.min(r_mu_list)], marker=r'$\ast$', color='w', s=0.0001)
plt.scatter([np.max(r_D_list)], [np.max(r_mu_list)], marker=r'$\ast$', color='w', s=0.0001)
plt.xlabel(r'$r_D$', fontsize=20)
plt.xticks(fontsize=12)
plt.xscale("log", base=2)
plt.ylabel(r'$r_\mu=\mu_C/\mu_W$', fontsize=20)
plt.yscale("log", base=2)
plt.yticks(fontsize=20)
plt.title(r'$W^{\ast}_m$', fontsize=20)
plt.gca().xaxis.set_major_locator(ticker.LogLocator(base=2))
plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
plt.tight_layout()
plt.savefig("w_m.png", dpi=300)
plt.show()
# W^* m


# C^* p
plt.figure()
Z = c_p_norm_final
plt.pcolormesh(r_D_list, r_mu_list, Z.T)
plt.colorbar()
plt.scatter([0.43], [3.92], marker=r'$\ast$', color='r', s=200)
plt.scatter([np.min(r_D_list)], [np.min(r_mu_list)], marker=r'$\ast$', color='w', s=0.0001)
plt.scatter([np.max(r_D_list)], [np.max(r_mu_list)], marker=r'$\ast$', color='w', s=0.0001)
plt.xlabel(r'$r_D$', fontsize=20)
plt.xticks(fontsize=12)
plt.xscale("log", base=2)
plt.ylabel(r'$r_\mu=\mu_C/\mu_W$', fontsize=20)
plt.yscale("log", base=2)
plt.yticks(fontsize=20)
plt.title(r'$C^{\ast}_p$', fontsize=20)
plt.gca().xaxis.set_major_locator(ticker.LogLocator(base=2))
plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
plt.tight_layout()
plt.savefig("c_p.png", dpi=300)
plt.show()
# C^* p

# C^* m
plt.figure()
Z = c_m_norm_final
plt.pcolormesh(r_D_list, r_mu_list, Z.T)
plt.colorbar()
plt.scatter([0.43], [3.92], marker=r'$\ast$', color='r', s=200)
plt.scatter([np.min(r_D_list)], [np.min(r_mu_list)], marker=r'$\ast$', color='w', s=0.0001)
plt.scatter([np.max(r_D_list)], [np.max(r_mu_list)], marker=r'$\ast$', color='w', s=0.0001)
plt.xlabel(r'$r_D$', fontsize=20)
plt.xticks(fontsize=12)
plt.xscale("log", base=2)
plt.ylabel(r'$r_\mu=\mu_C/\mu_W$', fontsize=20)
plt.yscale("log", base=2)
plt.yticks(fontsize=20)
plt.title(r'$C^{\ast}_m$', fontsize=20)
plt.gca().xaxis.set_major_locator(ticker.LogLocator(base=2))
plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
plt.tight_layout()
plt.savefig("c_m.png", dpi=300)
plt.show()
# C^* m



plt.figure()
Z = w_m_norm_final / w_p_norm_final
plt.pcolormesh(r_D_list, r_mu_list, Z.T)
plt.colorbar(label=r'$W^{\ast}_m/W^{\ast}_p$')
plt.scatter([0.43], [3.92], marker=r'$\ast$', color='r', s=200)
plt.scatter([np.min(r_D_list)], [np.min(r_mu_list)], marker=r'$\ast$', color='w', s=0.0001)
plt.scatter([np.max(r_D_list)], [np.max(r_mu_list)], marker=r'$\ast$', color='w', s=0.0001)
plt.xlabel(r'$r_D$', fontsize=20)
plt.xticks(fontsize=12)
plt.xscale("log", base=2)
plt.ylabel(r'$r_\mu=\mu_C/\mu_W$', fontsize=20)
plt.yscale("log", base=2)
plt.yticks(fontsize=20)
plt.title(r'$W^{\ast}_m/W^{\ast}_p$', fontsize=20)
plt.gca().xaxis.set_major_locator(ticker.LogLocator(base=2))
plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
plt.tight_layout()
plt.savefig("w_m_to_w_p.png", dpi=300)
plt.show()


plt.figure()
Z = c_m_norm_final / c_p_norm_final
plt.pcolormesh(r_D_list, r_mu_list, Z.T)
plt.colorbar(label=r'$C^{\ast}_m/C^{\ast}_p$')
plt.scatter([0.43], [3.92], marker=r'$\ast$', color='r', s=200)
plt.scatter([np.min(r_D_list)], [np.min(r_mu_list)], marker=r'$\ast$', color='w', s=0.0001)
plt.scatter([np.max(r_D_list)], [np.max(r_mu_list)], marker=r'$\ast$', color='w', s=0.0001)
plt.xlabel(r'$r_D$', fontsize=20)
plt.xticks(fontsize=12)
plt.xscale("log", base=2)
plt.ylabel(r'$r_\mu=\mu_C/\mu_W$', fontsize=20)
plt.yscale("log", base=2)
plt.yticks(fontsize=20)
plt.title(r'$C^{\ast}_m/C^{\ast}_p$', fontsize=20)
plt.gca().xaxis.set_major_locator(ticker.LogLocator(base=2))
plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
plt.tight_layout()
plt.savefig("c_m_to_c_p.png", dpi=300)
plt.show()


plt.figure()
Z = wd_p_perc_final
plt.pcolormesh(r_D_list, r_mu_list, Z.T)
plt.colorbar(label=r'$W_D$' +' frac (pure)')
plt.scatter([0.43], [3.92], marker=r'$\ast$', color='r', s=200)
plt.scatter([np.min(r_D_list)], [np.min(r_mu_list)], marker=r'$\ast$', color='w', s=0.0001)
plt.scatter([np.max(r_D_list)], [np.max(r_mu_list)], marker=r'$\ast$', color='w', s=0.0001)
plt.xlabel(r'$r_D$', fontsize=20)
plt.xticks(fontsize=12)
plt.xscale("log", base=2)
plt.ylabel(r'$r_\mu=\mu_C/\mu_W$', fontsize=20)
plt.yscale("log", base=2)
plt.yticks(fontsize=20)
plt.title(r'$W_D$' +' frac (pure)', fontsize=20)
plt.gca().xaxis.set_major_locator(ticker.LogLocator(base=2))
plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
plt.tight_layout()
plt.savefig("w_d_frac_p.png", dpi=300)
plt.show()




plt.figure()
Z = wd_m_perc_final
plt.pcolormesh(r_D_list, r_mu_list, Z.T)
plt.colorbar(label=r'$W_D$' +' frac (mixed)')
plt.scatter([0.43], [3.92], marker=r'$\ast$', color='r', s=200)
plt.scatter([np.min(r_D_list)], [np.min(r_mu_list)], marker=r'$\ast$', color='w', s=0.0001)
plt.scatter([np.max(r_D_list)], [np.max(r_mu_list)], marker=r'$\ast$', color='w', s=0.0001)
plt.xlabel(r'$r_D$', fontsize=20)
plt.xticks(fontsize=12)
plt.xscale("log", base=2)
plt.ylabel(r'$r_\mu=\mu_C/\mu_W$', fontsize=20)
plt.yscale("log", base=2)
plt.yticks(fontsize=20)
plt.title(r'$W_D$' +' frac (mixed)', fontsize=20)
plt.gca().xaxis.set_major_locator(ticker.LogLocator(base=2))
plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
plt.tight_layout()
plt.savefig("w_d_frac_m.png", dpi=300)
plt.show()