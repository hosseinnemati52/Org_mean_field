#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 13:14:51 2025

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
from matplotlib.lines import Line2D



def fit_paraboloid(
        x, y, z,
        sigma=None,          # 1‑σ uncertainties (same shape as z)  – OR –
        weights=None,        # user‑supplied weights  w_i  (same shape as z)
        mask=None,           # bool array that flags valid points (same shape as z)
        return_surface=False,
        compute_vertex=True,
        return_cov=False):
    """
    Weighted fit of  z = a x² + b y² + c x y + d x + e y + f.

    Parameters
    ----------
    x, y, z : array‑like
        • Regular grid   →   2‑D arrays from np.meshgrid and a matching z‑matrix.
        • Scattered data →   1‑D vectors of equal length.
    sigma : array‑like, optional
        1‑σ error bars on each z value.   Weight = 1 / sigma**2.
    weights : array‑like, optional
        Direct weights  w_i  (e.g. inverse variances).  Ignored if *sigma* given.
    mask : bool array, optional
        True where a point is valid.  NaNs are always excluded automatically.
    return_surface : bool, default False
        When x & y are 2‑D, also return the fitted surface  z_fit (same shape).
    compute_vertex : bool, default True
        Compute the extremum (x_v, y_v, z_v).
    return_cov : bool, default False
        Also return the 6×6 covariance matrix of the coefficients.

    Returns
    -------
    beta   : ndarray (6,)          – [a, b, c, d, e, f]
    vertex : (x_v, y_v, z_v) or None
    rms    : float                 – *un‑weighted* RMS of residuals
    chi2   : float                 – Σ w_i · r_i²            (always returned)
    cov    : ndarray (6,6) or None – only if return_cov=True
    z_fit  : ndarray               – only if return_surface=True & grid input
    """

    # -------------------- flatten & clean -------------------------------------
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    if sigma is not None and weights is not None:
        raise ValueError("Pass EITHER sigma OR weights, not both")

    if mask is None:
        mask = ~np.isnan(z)
    else:
        mask = mask & ~np.isnan(z)

    X = x.ravel()[mask.ravel()]
    Y = y.ravel()[mask.ravel()]
    Z = z.ravel()[mask.ravel()]

    # -------------------- weights --------------------------------------------
    if sigma is not None:
        W = 1.0 / np.asarray(sigma).ravel()[mask.ravel()]**2
    elif weights is not None:
        W = np.asarray(weights).ravel()[mask.ravel()]
    else:
        W = np.ones_like(Z)

    sqrtW = np.sqrt(W)                      # for easy row‑scaling

    # -------------------- design matrix --------------------------------------
    A = np.column_stack([X**2, Y**2, X*Y, X, Y, np.ones_like(X)])

    # Weighted least‑squares:  solve (√W A) β  ≈  (√W z)
    Aw = A * sqrtW[:, None]
    Zw = Z * sqrtW

    beta, *_ = np.linalg.lstsq(Aw, Zw, rcond=None)
    a, b, c, d, e, f = beta

    # -------------------- residual metrics -----------------------------------
    r = Z - (a*X**2 + b*Y**2 + c*X*Y + d*X + e*Y + f)   # residuals
    rms  = np.sqrt(np.mean(r**2))
    chi2 = np.sum(W * r**2)

    # -------------------- optional covariance --------------------------------
    cov = None
    if return_cov:
        # Var(β) ≈ (Aᵀ W A)⁻¹
        ATA_inv = np.linalg.inv(A.T @ (W[:, None] * A))
        cov = ATA_inv

    # -------------------- vertex (extremum) ----------------------------------
    vertex = None
    if compute_vertex:
        H = np.array([[2*a, c],
                      [c  , 2*b]])
        xv, yv = np.linalg.solve(H, -np.array([d, e]))
        zv = a*xv**2 + b*yv**2 + c*xv*yv + d*xv + e*yv + f
        vertex = (xv, yv, zv)

    # -------------------- optional reconstructed surface ---------------------
    z_fit = None
    if return_surface and x.ndim == 2:
        z_fit = (a*x**2 + b*y**2 + c*x*y + d*x + e*y + f)

    # -------------------- pack results ---------------------------------------
    out = [beta, vertex, rms, chi2]
    if return_cov:
        out.append(cov)
    if return_surface and x.ndim == 2:
        out.append(z_fit)
    return tuple(out)

n_sets = 200

s_list = np.loadtxt("set_1/s_list.csv", delimiter=',')
v_list = np.loadtxt("set_1/v_list.csv", delimiter=',')

cost_1 = np.loadtxt('set_1/cost.csv', delimiter=',')





cost_dict = dict()

v_min_list=[]
s_min_list=[]

temp_v_min_list = []
temp_s_min_list = []

cost_miv_vals = []

bunch_size = 10

for set_c in range(n_sets):
    foldar_name = "set_"+str(set_c+1)
    cost_indiv = np.loadtxt(foldar_name+'/cost.csv', delimiter=',')
    cost_dict[set_c] = cost_indiv.copy()
    
    
    if (set_c+1)%bunch_size==0:
        
        cost_avg_temp = 0.0 * cost_1.copy()
        for j in range(bunch_size):
            cost_avg_temp = cost_avg_temp + cost_dict[set_c-j]/bunch_size
            
        min_value = np.nanmin(cost_avg_temp)
        flat_index = np.nanargmin(cost_avg_temp)
        row, col = np.unravel_index(flat_index, cost_avg_temp.shape)
        
        cost_miv_vals.append(np.nanmin(cost_avg_temp))
        
        # temp_v_min_list.append(v_list[col])
        # temp_s_min_list.append(s_list[row])
        
        # v_min_list.append(np.mean(temp_v_min_list))
        # s_min_list.append(np.mean(temp_s_min_list))
        

        v_min_list.append(v_list[col])
        s_min_list.append(s_list[row])
        
        # del temp_v_min_list
        # del temp_s_min_list
        del cost_avg_temp
        # temp_v_min_list = []
        # temp_s_min_list = []
        cost_avg_temp = 0.0 * cost_1
        
    # elif (set_c+1)%bunch_size==1:
    #     temp_v_min_list.append(v_list[col])
    #     temp_s_min_list.append(s_list[row])
    # else:
    #     temp_v_min_list.append(v_list[col])
    #     temp_s_min_list.append(s_list[row])
        
        
    

cost_avg = 0.0 * cost_1
cost_err = 0.0 * cost_1

for set_c in range(n_sets):
    cost_avg = cost_avg + (cost_dict[set_c])/n_sets
    
for set_c in range(n_sets):
    cost_err = cost_err + ((cost_dict[set_c]-cost_avg)**2)/(n_sets-1) # variance

cost_err = cost_err**0.5 #STD

cost_err = cost_err / np.sqrt(n_sets-1) # SEM

cost=cost_avg.copy()
# cost=cost_err.copy()

min_value = np.nanmin(cost)
flat_index = np.nanargmin(cost)
row, col = np.unravel_index(flat_index, cost.shape)
v_val_min_cost = v_list[col]
s_val_min_cost = s_list[row]


v_val_min_cost_ellipse = np.mean(v_min_list)
s_val_min_cost_ellipse = np.mean(s_min_list)

C = np.cov(v_min_list, s_min_list)
chi2 = 5.99                    # 95 % confidence

U, s, _ = np.linalg.svd(C)
r = np.sqrt(chi2) * np.sqrt(s)  # semi‑axes lengths
theta = np.linspace(0, 2*np.pi, 200)
ellipse = (U @ np.diag(r) @
           np.vstack([np.cos(theta), np.sin(theta)])).T
ellipse[:,0] += v_val_min_cost_ellipse
ellipse[:,1] += s_val_min_cost_ellipse

# plot


fitting_cost = cost

# v_list = v_list[col-domain_v:col+domain_v+1]
# s_list = s_list[row-domain_s:row+domain_s+1]

V, S = np.meshgrid(v_list, s_list)

contour_levels_cost = np.log10(np.linspace(np.nanmin(cost), 1.05*np.nanmin(cost), 2))
plt.figure()
plt.pcolormesh(V, S,np.log10(fitting_cost) , shading='auto', cmap='viridis')
plt.xlabel(r'$\tilde{v}_{\mathrm{S}}$', fontsize=15)
plt.ylabel(r'$\tilde{S}$', fontsize=15)
plt.colorbar(label="log10(cost)")
# contours = plt.contour(V, S, np.log10(cost), levels=contour_levels_cost, colors='w', linewidths=1)
contours = plt.contour(V, S, np.log10(cost), levels=contour_levels_cost, colors='w', linewidths=2, label=['min'])
# dv = v_list[1] - v_list[0]
# ds = s_list[1] - s_list[0]
# grad_s, grad_v = np.gradient(np.log10(cost), ds, dv)  # note: axes might be flipped depending on your meshgrid
# g_v = grad_v[col, row]  # ∂f/∂v
# g_s = grad_s[col, row]
# m_max = g_v/g_s
# m_min = -1/m_max

# plt.arrow(v_val_min_cost, s_val_min_cost, 0.99, 0.11, head_width=0.05, head_length=0.05, fc='r', ec='r')

# plt.quiver(v_val_min_cost, s_val_min_cost, dv, ds, color='white', scale=10, width=0.1, label='Gradient')
# plt.quiver(v_val_min_cost, s_val_min_cost, -dv, -ds, color='red', scale=10, width=0.005, label='Steepest Descent')

# # Optionally add labels to the contour lines
# plt.clabel(contours, inline=True, fontsize=8)
# contours = plt.contour(V, S, phase_quantity_3, levels=contour_levels_g, colors='r', linewidths=1)
# Optionally add labels to the contour lines
# plt.clabel(contours, inline=True, fontsize=8)
# plt.scatter([v_val_min_cost],[s_val_min_cost], marker='*', color='r')
v_err = np.max([v_list[1]-v_list[0], np.std(v_min_list)])
s_err = np.max([s_list[1]-s_list[0], np.std(s_min_list)])
# plt.errorbar([v_val_min_cost],[s_val_min_cost], xerr=v_err, yerr=s_err, marker='*', color='r')
# plt.scatter(v_min_list,s_min_list,  marker='*', color='r')
min_samples = plt.scatter(v_min_list,s_min_list, color='r', s=20, zorder=9, marker='*')

plt.plot(ellipse[:,0], ellipse[:,1], color='w', linewidth=2)
min_overal = plt.scatter([v_val_min_cost],[s_val_min_cost], color='y', s=9, zorder=10)
#plt.scatter([10.76],[19.127], color='w', s=3)

# fdsadsadsaf
# Suppose you already have: X, Y, Z  (2‑D) and matching error bars SIGMA
cost=cost_avg.copy()

min_value = np.nanmin(cost)
flat_index = np.nanargmin(cost)
row, col = np.unravel_index(flat_index, cost.shape)
v_val_min_cost = v_list[col]
s_val_min_cost = s_list[row]

domain_s = 4
domain_v = 4
fitting_cost = cost[row-domain_s:row+domain_s+1, col-domain_v:col+domain_v+1]

X = V[row-domain_s:row+domain_s+1, col-domain_v:col+domain_v+1]
Y = S[row-domain_s:row+domain_s+1, col-domain_v:col+domain_v+1]
Z = cost_avg[row-domain_s:row+domain_s+1, col-domain_v:col+domain_v+1]
SIGMA = cost_err[row-domain_s:row+domain_s+1, col-domain_v:col+domain_v+1]

beta, vertex, rms, chi2, z_cov, Z_fit = fit_paraboloid(
        X, Y, Z,
        sigma=SIGMA,
        return_cov=True,
        return_surface=True)
print("Coefficients:", beta)
print("Chi² / dof:", chi2 / (Z.size - 6))
print("Vertex:", vertex)

Z_vv = 2 * beta[0]
Z_ss = 2 * beta[1]
Z_sv =  beta[2]

# Z_vv = (cost[col-1, row] - 2 *cost[col,row] + cost[col+1,row]) / (dv**2)
# Z_ss = (cost[col, row-1] - 2 *cost[col,row] + cost[col,row+1]) / (ds**2)
# Z_sv =  (cost[col+1,row+1]+cost[col-1,row-1]-cost[col-1,row+1]-cost[col+1,row-1]) / (4*dv*ds)

H_mat = np.array([[Z_vv, Z_sv],[Z_sv, Z_ss]])

eig_vals, eig_vecs = np.linalg.eig(H_mat)
eigvec_0 = eig_vecs[:, 0]
eigvec_1 = eig_vecs[:, 1]

L = 0.2

v_vec_high_inc = v_val_min_cost + L * np.array([-1,1]) * eigvec_0[0]
s_vec_high_inc = s_val_min_cost + L * np.array([-1,1]) * eigvec_0[1]

v_vec_low_inc = v_val_min_cost + 3*L * np.array([-1,1]) * eigvec_1[0]
s_vec_low_inc = s_val_min_cost + 3*L * np.array([-1,1]) * eigvec_1[1]

plt.plot(v_vec_high_inc, s_vec_high_inc, color='r', linestyle='--', linewidth=2)
plt.plot(v_vec_low_inc, s_vec_low_inc, color='b', linestyle='--', linewidth=2)




white_line = Line2D([0], [0], color='w', lw=2, linestyle='--')
w_line = Line2D([0], [0], color='w', lw=2)
r_line = Line2D([0], [0], color='r', lw=2, linestyle='--')
b_line = Line2D([0], [0], color='b', lw=2, linestyle='--')
legend = plt.legend([white_line, w_line, r_line, b_line, min_samples, min_overal], ['min(cost) + 5 %', '95 % confidence', 'highest curvature', 'lowest curvature', 'min(cost), samples', 'min(cost), overal'], fontsize=8)# plt.xlim((10.3,11.5))
frame = legend.get_frame()
frame.set_facecolor('lightgray')  # Background color
frame.set_edgecolor('black')      # Border color
frame.set_alpha(0.5)   
plt.ylim((18.5,20.5))


plt.gca().set_aspect('equal')
# plt.gca().set_aspect('equal'); plt.show()
# plt.errorbar([10.8],[19.2], xerr=0.1, yerr=0.2, marker='*', color='r')
# plt.xscale('log')
# plt.yscale('log')
plt.savefig('cost_fitting_heatmap.png', dpi=300)
# plot

