# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:20:47 2025

@author: Nemat002
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

class single_org_sol_class:
    pass

class params_class:
    K_C = 0

class solution_class:
    time_tilde = 0

class gd_hist_class:
    cost_hist = []
    
    
def sol_population_plotter(par, sol):
    
    
    plt.figure()
    
    
    C_mix_exp = np.loadtxt("exp_data"+"/"+"C_bar_mix_overal.csv", delimiter=',')
    WT_mix_exp = np.loadtxt("exp_data"+"/"+"WT_bar_mix_overal.csv", delimiter=',')
    C_pure_exp = np.loadtxt("exp_data"+"/"+"overal_C_pure.csv", delimiter=',')
    WT_pure_exp = np.loadtxt("exp_data"+"/"+"overal_WT_pure.csv", delimiter=',')
    
    
    #C_mix
    # plt.scatter(C_mix_exp[0,:], C_mix_exp[1,:], label='exp_C_mix')
    plt.errorbar(C_mix_exp[0,:], C_mix_exp[1,:], yerr=C_mix_exp[2,:], fmt='s', color='g', ecolor='g', capsize=2, label='mixed C (exp)', markerfacecolor='none', zorder=10)
    
    #WT_mix
    # plt.scatter(WT_mix_exp[0,:], WT_mix_exp[1,:], label='exp_WT_mix')
    plt.errorbar(WT_mix_exp[0,:], WT_mix_exp[1,:], yerr=WT_mix_exp[2,:], fmt='s', color='m', ecolor='m', capsize=2, label='mixed WT (exp)', markerfacecolor='none', zorder=10)
    
    #C_pure
    # plt.scatter(C_pure_exp[0,:], C_pure_exp[1,:], label='exp_C_pure')
    plt.errorbar(C_pure_exp[0,:], C_pure_exp[1,:], yerr=C_pure_exp[2,:], fmt='o', color='g', ecolor='g', capsize=2, label='pure C (exp)', zorder=10)
    
    #WT_pure
    # plt.scatter(WT_pure_exp[0,:], WT_pure_exp[1,:], label='exp_WT_pure')
    plt.errorbar(WT_pure_exp[0,:], WT_pure_exp[1,:], yerr=WT_pure_exp[2,:], fmt='o', color='m', ecolor='m', capsize=2, label='pure WT (exp)', zorder=10)
        
    
        
    plt.plot(sol.time_tilde/par.beta_wt, sol.pure_w_sol.n_w_tot, label='pure WT (model)', color = 'm', linestyle='--')
    plt.plot(sol.time_tilde/par.beta_wt, sol.pure_c_sol.n_c_tot, label='pure C (model)', color = 'g', linestyle='--')

    n_org = par.n_org
    
    n_w_mix_mat = np.zeros((n_org, sol.L))
    n_c_mix_mat = np.zeros((n_org, sol.L))
    for org_c in range(n_org):
        n_w_mix_mat[org_c,:] = sol.mixed_sol[org_c].n_w_tot / sol.mixed_sol[org_c].n_w_tot[0]
        n_c_mix_mat[org_c,:] = sol.mixed_sol[org_c].n_c_tot / sol.mixed_sol[org_c].n_c_tot[0]
    
    time = sol.time_tilde/par.beta_wt
    y_avg = np.mean(n_w_mix_mat, axis=0)
    y_err = np.std(n_w_mix_mat, axis=0)/np.sqrt(n_org)
    # plt.plot(time, y_avg, color = 'm', linestyle='--')
    plt.errorbar(time, y_avg, yerr=y_err, capsize=0, label='mixed WT (model)', color = 'm', alpha=0.6)
    
    y_avg = np.mean(n_c_mix_mat, axis=0)
    y_err = np.std(n_c_mix_mat, axis=0)/np.sqrt(n_org)
    # plt.plot(time, y_avg)
    plt.errorbar(time, y_avg, yerr=y_err, capsize=0, label='mixed C (model)', color = 'g', alpha=0.6)
    
    plt.xlabel('time(h)', fontsize=15)
    plt.ylabel(r'$\langle N(t)/N(0) \rangle$', fontsize=15)
    # plt.grid()
    plt.xlim((np.min(time), np.max(time)))
    plt.xticks( fontsize=15)
    plt.yticks(fontsize=15)
    plt.yscale("log")
    plt.gca().set_aspect(30)
    # plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.legend()
    plt.tight_layout()
    plt.savefig('mean_field_four_pops.PNG', dpi=300)
    
    # plot_data = np.zeros((5, len(time)))
    # plt.figure()
    # WT_mix_model =  np.log(np.mean(n_w_mix_mat, axis=0))
    # diff = (WT_mix_model[1:]- WT_mix_model[:-1]) / ( time[1:]- time[:-1] )
    # plt.plot(time[:-1],diff)
    
    return 
    
def eq_conc_wt(x, par):
    # return (par.S / x - 1 / (x + par.K_P) - par.R_0_d - par.beta_wt/par.muF_0)
    return (par.S / x - 1 / (x + par.K_P) - par.R_0_d - par.R_0_d*par.frac/(1-par.frac))

# Optional: derivative
def eq_conc_wt_prime(x, par):
    return (- par.S / x**2 + 1 / (x + par.K_P)**2)


def cost_calc_detail(exp_data, model_data):
    
    n_row, n_col = np.shape(exp_data)
    cost_detail = 0.0
    
    # exp_data[2,0] = 1e-8
    
    log_err = (exp_data[2,:]/exp_data[1,:])
    # log_err_relative = log_err/np.log(exp_data[1,:])
    
    weights = (1/log_err)**2
    # weights = np.ones(len(weights))
    weights[0] = 0
    weights = weights/np.sum(weights)
    
    final_exclude = 0
    unweighted_cost_list = np.zeros(n_col-final_exclude)
    unweighted_cost_list[0] = 0
    for i in range(1,n_col-final_exclude):
        
        x_exp_val = exp_data[0,i]
        y_exp_val = exp_data[1,i]
        
        closest_arg = np.argmin(np.abs(model_data[0,:]-x_exp_val))
        y_model_val = model_data[1,closest_arg]

        # cost_detail += weights[i] * ((np.log(y_exp_val)-np.log(y_model_val))/np.log(y_exp_val))**2 
        unweighted_cost_list[i] = ((np.log(y_exp_val)-np.log(y_model_val)))**2 
        cost_detail += weights[i] * unweighted_cost_list[i]
    
    return cost_detail, unweighted_cost_list

def cost_calc(F_0, s, c_w, c_c, plot_switch):
    
    def cost_calc_detail(exp_data, model_data):
        
        n_row, n_col = np.shape(exp_data)
        cost_detail = 0.0
        
        # exp_data[2,0] = 1e-8
        
        log_err = (exp_data[2,:]/exp_data[1,:])
        # log_err_relative = log_err/np.log(exp_data[1,:])
        
        weights = (1/log_err)**2
        # weights = np.ones(len(weights))
        weights[0] = 0
        weights = weights/np.sum(weights)
        
        for i in range(1,n_col):
            
            x_exp_val = exp_data[0,i]
            y_exp_val = exp_data[1,i]
            
            closest_arg = np.argmin(np.abs(model_data[0,:]-x_exp_val))
            y_model_val = model_data[1,closest_arg]

            # cost_detail += weights[i] * ((np.log(y_exp_val)-np.log(y_model_val))/np.log(y_exp_val))**2 
            cost_detail += weights[i] * ((np.log(y_exp_val)-np.log(y_model_val)))**2 
        
        return cost_detail
    
    n_org = 100
    
    # n_w_0_list = np.random.randint(10,100,n_org)
    # n_w_0_list = np.ones(100)
    # n_c_0_list = np.linspace(1, 10, 100)
    # n_c_0_list = np.random.randint(10,100,n_org)
    
    mixed_sample_bank = np.loadtxt("mixed_sample_bank.csv", delimiter=',', dtype=int)
    size = np.shape(mixed_sample_bank)[0]
    sample_indices = np.random.randint(0,size,n_org)
    n_w_0_list = []
    n_c_0_list = []
    for sample_c in range(n_org):
        n_w_0_list.append(mixed_sample_bank[sample_indices[sample_c], 0])
        n_c_0_list.append(mixed_sample_bank[sample_indices[sample_c], 1])
    n_w_0_list = np.array(n_w_0_list)
    n_c_0_list = np.array(n_c_0_list)
    
    # plt.figure()
    # plt.hist(n_w_0_list, 30)
    
    # plt.figure()
    # plt.hist(n_c_0_list, 30)
    
    # shsdj
    
    
    init_w_ratio = n_w_0_list/(n_w_0_list+n_c_0_list)
    init_c_ratio = 1-init_w_ratio
    
    c_mix_growth = np.zeros(n_org)
    w_mix_growth = np.zeros(n_org)
    
    t_max = 70
    time = np.linspace(0, t_max, 500)
    L = len(time)
    dt = time[1]-time[0]
    
    arg_eval = np.argmin(np.abs(time-60))
    
    norm_n_w_p_mat = np.zeros((n_org,L))
    norm_n_c_p_mat = np.zeros((n_org,L))
    norm_n_w_m_mat = np.zeros((n_org,L))
    norm_n_c_m_mat = np.zeros((n_org,L))
    
    
    for j in range(len(n_w_0_list)):
    
        
        
        # f_0 = 0.06
        d_0 = 0.0 # death rate
        # s = 1
        
        # factor = .1
        # c_w = 0.3 * factor
        # c_c = c_w/.29411
        
        
        
        # n_w_p = np.zeros(L)
        # n_c_p = np.zeros(L)
        
        # n_w_m = np.zeros(L)
        # n_c_m = np.zeros(L)
        # n_t_m = np.zeros(L)
        # m   = np.zeros(L)
        
        
        
        # n_w_p = solve_w()
        
        # n_c_p = solve_c()
        
        # n_w_0_p = 1
        # n_c_0_p = 1
        # n_w_0_m = 1
        # n_c_0_m = 1
        
        n_w_0_p = n_w_0_list[j]
        n_c_0_p = n_c_0_list[j]
        n_w_0_m = n_w_0_list[j]
        n_c_0_m = n_c_0_list[j]
        
        # c0_w = s/c_w
        # c0_c = s/c_c
        c0_w =  (-1 + np.sqrt(1+4*s*F_0/c_w)) / (2 * F_0)
        c0_c =  (-1 + np.sqrt(1+4*s*F_0/c_c)) / (2 * F_0)
        weghted_c0 = (n_w_0_m/(n_w_0_m+n_c_0_m))*c0_w+(n_c_0_m/(n_w_0_m+n_c_0_m))*c0_c
        # weghted_c0 = c0_w
        # weghted_c0 = c0_c
        
        
        m_w_p, n_w_p, bbbbb = solve_both(F_0, s, c_w, c_c, L, d_0, dt, c0_w, n_w_0_p, 0.0)
        m_c_p, bbbbb, n_c_p = solve_both(F_0, s, c_w, c_c, L, d_0, dt, c0_c, 0.0, n_c_0_p)
        m_m, n_w_m, n_c_m = solve_both(F_0, s, c_w, c_c, L, d_0, dt, weghted_c0, n_w_0_m, n_c_0_m)
        # m_m, n_w_m, n_c_m = solve_both(1.0, n_w_0_list[j], n_c_0_list[j])
        # m_m, n_w_m, n_c_m = solve_both(  weghted_c0, n_w_0_list[j], n_c_0_list[j])
        
        # m_w_p, n_w_p, bbbbb = solve_both_chemostat(1.0, 1.0, 0.0)
        # m_c_p, bbbbb, n_c_p = solve_both_chemostat(1.0, 0.0, 1.0)
        # m_m, n_w_m, n_c_m = solve_both_chemostat(1.0, 1.0, 1.0)
        # m_m, n_w_m, n_c_m = solve_both_chemostat(1.0, n_w_0_list[j], n_c_0_list[j])
        
        norm_n_w_p_mat[j,:] = n_w_p/n_w_p[0]
        norm_n_c_p_mat[j,:] = n_c_p/n_c_p[0]
        norm_n_w_m_mat[j,:] = n_w_m/n_w_m[0]
        norm_n_c_m_mat[j,:] = n_c_m/n_c_m[0]
        
        
        
        c_mix_growth[j] = n_c_m[arg_eval]/n_c_m[0]
        w_mix_growth[j] = n_w_m[arg_eval]/n_w_m[0]
        
        # plt.figure()
        # plt.plot(time, m_w_p/n_w_p, label='WT_pure_conce')
        # plt.plot(time, m_c_p/n_c_p, label='C_pure_conce')
        # plt.plot(time, m_m/(n_w_m+n_c_m), label='mixed_conce')
        # plt.grid()
        # plt.legend()
        # plt.show()
        
        # plt.figure()
        # plt.plot(time, n_w_p/n_w_p[0], label='WT_pure')
        # plt.plot(time, n_c_p/n_c_p[0], label='C_pure')
        # plt.plot(time, n_w_m/n_w_m[0], label='WT_mix', linestyle='--', zorder=10)
        # # plt.plot(time, n_w_m/(n_w_m+n_c_m), label='WT_mix', linestyle='--', zorder=10)
        # plt.plot(time, n_c_m/n_c_m[0], label='C_mix', linestyle='--', zorder=10)
        # plt.scatter([60,60,60,60], [5.62,3.0,10.5, 16.5 ])
        # plt.grid(which='both')
        # plt.yscale("log")
        # # plt.ylim((1,200))
        # plt.legend()
        
        # sdzfsd
        
    
    C_mix_exp = np.loadtxt("exp_data"+"/"+"C_bar_mix_overal.csv", delimiter=',')
    WT_mix_exp = np.loadtxt("exp_data"+"/"+"WT_bar_mix_overal.csv", delimiter=',')
    C_pure_exp = np.loadtxt("exp_data"+"/"+"overal_C_pure.csv", delimiter=',')
    WT_pure_exp = np.loadtxt("exp_data"+"/"+"overal_WT_pure.csv", delimiter=',')
    
    C_mix_model   = np.zeros((3,L))
    WT_mix_model  = np.zeros((3,L))
    C_pure_model  = np.zeros((3,L))
    WT_pure_model = np.zeros((3,L))
    
    C_mix_model[0,:] = time.copy()
    WT_mix_model[0,:] = time.copy()
    C_pure_model[0,:] = time.copy()
    WT_pure_model[0,:] = time.copy()
    
    C_mix_model[1,:] = np.mean(norm_n_c_m_mat, axis=0).copy()
    WT_mix_model[1,:] = np.mean(norm_n_w_m_mat, axis=0).copy()
    C_pure_model[1,:] = np.mean(norm_n_c_p_mat, axis=0).copy()
    WT_pure_model[1,:] = np.mean(norm_n_w_p_mat, axis=0).copy()
    
    C_mix_model[2,:] = np.std(norm_n_c_m_mat, axis=0).copy()/np.sqrt(n_org)
    WT_mix_model[2,:] = np.std(norm_n_w_m_mat, axis=0).copy()/np.sqrt(n_org)
    C_pure_model[2,:] = np.std(norm_n_c_p_mat, axis=0).copy()/np.sqrt(n_org)
    WT_pure_model[2,:] = np.std(norm_n_w_p_mat, axis=0).copy()/np.sqrt(n_org)
    
    if plot_switch :
        plt.figure()
        plt.errorbar(time, WT_pure_model[1,:], yerr = WT_pure_model[2,:] , label='WT_pure')
        plt.errorbar(time, C_pure_model[1,:], yerr = C_pure_model[2,:] , label='C_pure')
        plt.errorbar(time, WT_mix_model[1,:], yerr = WT_mix_model[2,:] , label='WT_mix')
        plt.errorbar(time, C_mix_model[1,:], yerr = C_mix_model[2,:] , label='C_mix')
        
        #C_mix
        plt.scatter(C_mix_exp[0,:], C_mix_exp[1,:], label='exp_C_mix')
        plt.errorbar(C_mix_exp[0,:], C_mix_exp[1,:], yerr=C_mix_exp[2,:], fmt='none')
        
        #WT_mix
        plt.scatter(WT_mix_exp[0,:], WT_mix_exp[1,:], label='exp_WT_mix')
        plt.errorbar(WT_mix_exp[0,:], WT_mix_exp[1,:], yerr=WT_mix_exp[2,:], fmt='none')
        
        #C_pure
        plt.scatter(C_pure_exp[0,:], C_pure_exp[1,:], label='exp_C_pure')
        plt.errorbar(C_pure_exp[0,:], C_pure_exp[1,:], yerr=C_pure_exp[2,:], fmt='none')
        
        #WT_pure
        plt.scatter(WT_pure_exp[0,:], WT_pure_exp[1,:], label='exp_WT_pure')
        plt.errorbar(WT_pure_exp[0,:], WT_pure_exp[1,:], yerr=WT_pure_exp[2,:], fmt='none')
        
        
        plt.xlabel('time(h)')
        plt.ylabel('Normalized Number')
        plt.grid()
        plt.yscale("log")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig('mean_field_norm.PNG', dpi=300)
        
        # plt.scatter([60,60,60,60], [5.62,3.0,10.5, 16.5 ], zorder=10)
        # plt.errorbar([60,60,60,60], [5.62,3.0,10.5, 16.5 ], yerr = [1.0, 0.5, 1.0, 4.0], zorder=10, linestyle='none')
        # plt.grid(which='both')
        # plt.yscale("log")
        # # plt.ylim((1,200))
        # plt.legend()
        plt.show()
    
    cost = 0.0
    # cost = cost + (np.log(np.mean(norm_n_w_p_mat, axis=0)[-1]) - np.log(final_w_pure))**2
    # cost = cost + (np.log(np.mean(norm_n_w_m_mat, axis=0)[-1]) - np.log(final_w_mix))**2
    # cost = cost + (np.log(np.mean(norm_n_c_p_mat, axis=0)[-1]) - np.log(final_c_pure))**2
    # cost = cost + (np.log(np.mean(norm_n_c_m_mat, axis=0)[-1]) - np.log(final_c_mix))**2
    
    cost = cost + 0.25*cost_calc_detail(C_mix_exp, C_mix_model)
    cost = cost + 0.25*cost_calc_detail(C_pure_exp, C_pure_model)
    cost = cost + 0.25*cost_calc_detail(WT_mix_exp, WT_mix_model)
    cost = cost + 0.25*cost_calc_detail(WT_pure_exp, WT_pure_model)
    
    
    
    
    init_compos_data = np.loadtxt("exp_data"+"/"+"init_compos_data_exp.csv", delimiter=',')
    
    plt.figure()
    plt.scatter(init_w_ratio, w_mix_growth, color='m', label= 'model')
    plt.scatter(init_compos_data[0,:], init_compos_data[1,:], color='m', marker='*', label= 'exp')
    plt.xlabel('initial WT percentage')
    plt.ylabel('WT Normalized Number at t=60h')
    plt.legend()
    plt.tight_layout()
    plt.savefig('mean_field_WT_compos.PNG', dpi=300)
    
    
    plt.figure()
    plt.scatter(init_c_ratio, c_mix_growth, color='g', label='model')
    plt.scatter(init_compos_data[2,:], init_compos_data[3,:], color='g', marker='*', label='exp')
    plt.xlabel('initial C percentage')
    plt.ylabel('C Normalized Number at t=60h')
    plt.legend()
    plt.tight_layout()
    plt.savefig('mean_field_C_compos.PNG', dpi=300)
    
    sdfkjsf
    
    return cost

def eq_solver(sF, beta):
    consumption = (beta**2)/(sF-beta)
    return consumption

def nondim_pure_solver(L, n_0, M_0, S, K_factor, dt):
    
    n_pure =  np.zeros(L)
    M_pure =  np.zeros(L)
    
    n_pure[0] = n_0
    M_pure[0] = M_0
    
    for i in range(1,L):
        
        dn_dt =  M_pure[i-1] / (M_pure[i-1]/n_pure[i-1] + K_factor)
        
        dM_dt = S * n_pure[i-1] \
              - M_pure[i-1] / (M_pure[i-1]/n_pure[i-1] + K_factor)
        
        n_pure[i] = n_pure[i-1] + dt * dn_dt
        M_pure[i] = M_pure[i-1] + dt * dM_dt
    
    # plt.figure()
    # plt.plot(time, m/n_t_m, label='concentration')
    # plt.legend()
    # plt.show()
    
    return n_pure, M_pure

def nondim_mix_solver(L, n_0_wt_mix, n_0_c_mix, M_0_mix, S, K_W, K_C, dt, r):
    
    n_w =  np.zeros(L)
    n_c =  np.zeros(L)
    n_t =  np.zeros(L)
    
    M   =  np.zeros(L)
    
    n_w[0] = n_0_wt_mix
    n_c[0] = n_0_c_mix
    n_t[0] = n_w[0] + n_c[0]
    M[0] = M_0_mix
    
    for i in range(1,L):
        
        concentration = M[i-1] / n_t[i-1] 
        
        # r = 0.0003
        
        dn_w_dt =  n_w[i-1] * concentration / (concentration + K_W) - r * n_w[i-1] * n_c[i-1]
        dn_c_dt =  n_c[i-1] * concentration / (concentration + K_C)
        
        dM_dt = S * n_t[i-1] \
              - dn_w_dt \
              - dn_c_dt \
              # + 5*r * n_w[i-1] * n_c[i-1]
              
        
        n_w[i] = n_w[i-1] + dt * dn_w_dt
        n_c[i] = n_c[i-1] + dt * dn_c_dt
        n_t[i] = n_w[i] + n_c[i]
        M[i] = M[i-1] + dt * dM_dt
        
    return n_w, n_c, M

def nondim_total_solver(par, L, n_0_p, n_0_d, n_0_c, M_0, dt):
    
    n_p =  np.zeros(L)
    n_d =  np.zeros(L)
    n_c =  np.zeros(L)
    n_t =  np.zeros(L)
    
    M   =  np.zeros(L)
    
    n_p[0] = n_0_p
    n_d[0] = n_0_d
    n_c[0] = n_0_c
    n_t[0] = n_p[0] + n_d[0] + n_c[0]
    M[0] = M_0
    
    for i in range(1,L):
        
        concentration = M[i-1] / n_t[i-1] 
        
        # r = 0.0003
        
        dn_p_dt =  n_p[i-1] * concentration / (concentration + par.K_P) - (par.R_0_d + par.R_pd * n_c[i-1]) * n_p[i-1]
        dn_d_dt =  (par.R_0_d + par.R_pd * n_c[i-1]) * n_p[i-1] - par.R_a * n_c[i-1] * n_d[i-1]
        dn_c_dt =  n_c[i-1] * concentration / (concentration + par.K_C) + par.R_b * n_d[i-1] * n_c[i-1]
        # dn_c_dt =  n_c[i-1] * concentration / (concentration + par.K_C) + par.R_b * dn_d_dt * n_c[i-1]
        
        dM_dt = par.S * (n_p[i-1] + n_c[i-1]) \
              - n_p[i-1] * concentration / (concentration + par.K_P) \
              - n_c[i-1] * concentration / (concentration + par.K_C)
              # + 5*r * n_w[i-1] * n_c[i-1]
              
        
        n_p[i] = n_p[i-1] + dt * dn_p_dt
        n_d[i] = n_d[i-1] + dt * dn_d_dt
        n_c[i] = n_c[i-1] + dt * dn_c_dt
        n_t[i] = n_p[i] + n_d[i] + n_c[i]
        
        M[i] = M[i-1] + dt * dM_dt
        
    return n_p, n_d, n_c, M
    
def rho_solver(par, sol, rho_G1_0_0_w, rho_G1_0_0_c, n_0_d, M_0):
    
    L = sol.L
    dt = sol.dt
    
    n_w_g1     = np.zeros(L)
    n_w_s      = np.zeros(L)
    n_w_d      = np.zeros(L)
    n_w_apop   = np.zeros(L)
    n_w_tot    = np.zeros(L)
    
    n_c_g1     = np.zeros(L)
    n_c_s      = np.zeros(L)
    n_c_tot    = np.zeros(L)
    
    N_tot = np.zeros(L)
    
    M = np.zeros(L)
    
    rho_G1_W = np.zeros((L, sol.n_mesh_G1)) # each row is for a single time and all x's
    rho_G1_C = np.zeros((L, sol.n_mesh_G1)) # each row is for a single time and all x's
    rho_S_W  = np.zeros((L, sol.n_mesh_S)) # each row is for a single time and all x's
    rho_S_C  = np.zeros((L, sol.n_mesh_S)) # each row is for a single time and all x's
    
    ## initialization (t=0)
    # exp_coef_G1_w = (1+par.R_0_d)*(par.alpha_w+par.K_W)/(par.F_w * par.alpha_w)
    # exp_coef_G1_c = par.gamma *(par.alpha_c+par.K_C)/(par.F_c * par.alpha_c)
    # exp_coef_S_w  = 1 / par.V_S
    # exp_coef_S_c  = par.gamma / par.V_S
    
    exp_coef_G1_w = par.m
    exp_coef_G1_c = par.q
    exp_coef_S_w  = par.m_prime
    exp_coef_S_c  = par.q_prime
    
    # mid_factor_w = np.exp(-exp_coef_G1_w * np.pi)* (par.alpha_w*par.F_w) / ((par.K_W+par.alpha_w)*par.V_S)
    # mid_factor_c = np.exp(-exp_coef_G1_c * np.pi)* (par.alpha_c*par.F_c) / ((par.K_C+par.alpha_c)*par.V_S)
    
    mid_factor_w = par.A
    mid_factor_c = par.B
    
    rho_G1_W[0,:] = rho_G1_0_0_w * np.exp( -exp_coef_G1_w * sol.x_G1)
    rho_G1_C[0,:] = rho_G1_0_0_c * np.exp( -exp_coef_G1_c * sol.x_G1)
    
    rho_S_W[0,:] = rho_G1_0_0_w * mid_factor_w * np.exp( -exp_coef_S_w * (sol.x_S - np.pi))
    rho_S_C[0,:] = rho_G1_0_0_c * mid_factor_c * np.exp( -exp_coef_S_c * (sol.x_S - np.pi))
    
    n_w_g1[0] = (np.sum(rho_G1_W[0,1:-1]) + 0.5 *rho_G1_W[0,0]+ 0.5 *rho_G1_W[0,-1] ) * sol.dx_G1
    n_w_s[0] =  (np.sum(rho_S_W[0,1:-1])  + 0.5 *rho_S_W[0,0]+  0.5 *rho_S_W[0,-1]  ) * sol.dx_S
    n_w_d[0] =  n_0_d
    n_w_apop[0] = 0.0
    n_w_tot[0] = n_w_g1[0] + n_w_s[0] + n_w_d[0]
    
    n_c_g1[0] = (np.sum(rho_G1_C[0,1:-1]) + 0.5 *rho_G1_C[0,0]+ 0.5 *rho_G1_C[0,-1] ) * sol.dx_G1
    n_c_s[0] =  (np.sum(rho_S_C[0,1:-1])  + 0.5 *rho_S_C[0,0]+  0.5 *rho_S_C[0,-1]  ) * sol.dx_S
    n_c_tot[0] =n_c_g1[0] + n_c_s[0]
    
    N_tot[0] = n_w_tot[0] + n_c_tot[0]
    
    M[0] = M_0
    ## initialization (t=0)
    
    # plt.figure()
    # plt.plot(sol.x_G1, rho_G1_W[0,:], label='WT')
    # plt.plot(sol.x_S, rho_S_W[0,:], label='WT')
    # plt.plot(sol.x_G1, rho_G1_C[0,:], label='C')
    # plt.plot(sol.x_S, rho_S_C[0,:], label='C')
    # plt.legend()
    
    # solving
    rho_G1_w_Xderiv = np.zeros(sol.n_mesh_G1)
    rho_S_w_Xderiv = np.zeros(sol.n_mesh_S)
    rho_G1_c_Xderiv = np.zeros(sol.n_mesh_G1)
    rho_S_c_Xderiv = np.zeros(sol.n_mesh_S)
    
    rho_G1_w_Tderiv = np.zeros(sol.n_mesh_G1)
    rho_S_w_Tderiv = np.zeros(sol.n_mesh_S)
    rho_G1_c_Tderiv = np.zeros(sol.n_mesh_G1)
    rho_S_c_Tderiv = np.zeros(sol.n_mesh_S)
    
    concen = np.zeros(L)
    concen[0] = M[0] / N_tot[0]
    
    t_delay = 0.15
    n_ind_delay = int(t_delay/dt)
    
    for dt_c in range(1,L):
        
        # if dt_c > n_ind_delay:
            
        #     # V_G1_W = np.sum(par.F_w * concen[dt_c-1-n_ind_delay:dt_c-1] / (concen[dt_c-1-n_ind_delay:dt_c-1] + par.K_W))*dt/t_delay
        #     # V_G1_C = np.sum(par.F_c * concen[dt_c-1-n_ind_delay:dt_c-1] / (concen[dt_c-1-n_ind_delay:dt_c-1] + par.K_C))*dt/t_delay
        #     avg_conc_w = np.sum(concen[dt_c-1-n_ind_delay:dt_c-1])*dt / t_delay
        #     avg_conc_c = avg_conc_w
        # else:
        #     avg_conc_w  = np.sum(concen)*dt / t_delay
        #     avg_conc_w += par.alpha_w * (t_delay-sol.time_tilde[dt_c-1])/t_delay
        #     avg_conc_c = np.sum(concen)*dt / t_delay
        #     avg_conc_c += par.alpha_c * (t_delay-sol.time_tilde[dt_c-1])/t_delay
        
        avg_conc_w = concen[dt_c-1]
        avg_conc_c = avg_conc_w
            
        # V_G1_W = par.F_w * avg_conc_w / (avg_conc_w + par.K_W)
        # V_G1_C = par.F_c * avg_conc_c / (avg_conc_c + par.K_C)
        
        V_G1_W =  par.MU_W * avg_conc_w
        V_G1_C =  par.V_0 + par.MU_C * avg_conc_c
        
        # I write all x derivatves backward because the flow is to the right
        
        # these ones update with continuity eq (BC)
        # if dt_c==1:
        #     rho_G1_W[dt_c, 0] = 2 * (par.V_S / V_G1_W) * rho_S_W[dt_c-1, -1]
        #     rho_G1_C[dt_c, 0] = 2 * (par.V_S / V_G1_C) * rho_S_C[dt_c-1, -1]
        #     rho_S_W[dt_c, 0] =  (V_G1_W / par.V_S) * rho_G1_W[dt_c-1, -1]
        #     rho_S_C[dt_c, 0] =  (V_G1_C / par.V_S) * rho_G1_C[dt_c-1, -1]
        # else:
        #     rho_G1_W[dt_c, 0] = rho_G1_W[dt_c-1, 0] + 2 * (par.V_S / V_G1_W) * (rho_S_W[dt_c-1, -1]-rho_S_W[dt_c-2, -1])
        #     rho_G1_C[dt_c, 0] = rho_G1_C[dt_c-1, 0] + 2 * (par.V_S / V_G1_C) * (rho_S_C[dt_c-1, -1]-rho_S_C[dt_c-2, -1])
        #     rho_S_W[dt_c, 0] =  rho_S_W[dt_c-1, 0] + (V_G1_W / par.V_S) * (rho_G1_W[dt_c-1, -1] - rho_G1_W[dt_c-2, -1])
        #     rho_S_C[dt_c, 0] =  rho_S_C[dt_c-1, 0] + (V_G1_C / par.V_S) * (rho_G1_C[dt_c-1, -1] - rho_G1_C[dt_c-2, -1])
        
        rho_G1_W[dt_c, 0] = 2 * (par.V_S / V_G1_W) * rho_S_W[dt_c-1, -1]
        rho_G1_C[dt_c, 0] = 2 * (par.V_S / V_G1_C) * rho_S_C[dt_c-1, -1]
        rho_S_W[dt_c, 0] =  (V_G1_W / par.V_S) * rho_G1_W[dt_c-1, -1]
        rho_S_C[dt_c, 0] =  (V_G1_C / par.V_S) * rho_G1_C[dt_c-1, -1]
        
        
        # rho_G1_W[dt_c, 0] = 2 * (par.V_S / V_G1_W) * (rho_S_W[dt_c-1, -1] + 0.5 *(rho_S_W[dt_c-1, -1]-rho_S_W[dt_c-1, -2])  ) + 0.5 * (rho_G1_W[dt_c-1, 1] - rho_G1_W[dt_c-1, 0])
        # rho_G1_C[dt_c, 0] = 2 * (par.V_S / V_G1_C) * rho_S_C[dt_c-1, -1]
        # rho_S_W[dt_c, 0] =  (V_G1_W / par.V_S) * rho_G1_W[dt_c-1, -1] 
        # rho_S_C[dt_c, 0] =  (V_G1_C / par.V_S) * rho_G1_C[dt_c-1, -1]
        # the others update based on the equations
        
        
        #Xderiv: WT in G1
        rho_G1_w_Xderiv[1:] =   (rho_G1_W[dt_c-1, 1:] - rho_G1_W[dt_c-1, 0:-1])   /  sol.dx_G1 #backward
        #Xderiv: C in G1
        rho_G1_c_Xderiv[1:] =   (rho_G1_C[dt_c-1, 1:] - rho_G1_C[dt_c-1, 0:-1])   /  sol.dx_G1 #backward
        #Xderiv: WT in S
        rho_S_w_Xderiv[1:] =   (rho_S_W[dt_c-1, 1:] - rho_S_W[dt_c-1, 0:-1])      /  sol.dx_S #backwardd
        #Xderiv: C in S
        rho_S_c_Xderiv[1:] =   (rho_S_C[dt_c-1, 1:] - rho_S_C[dt_c-1, 0:-1])      /  sol.dx_S #backwardd
        
        
        
        # # fluxes
        # J_out_w = rho_S_w_Xderiv[-1]  * par.V_S
        # J_out_c = rho_S_c_Xderiv[-1]  * par.V_S
        # J_mid_w = rho_G1_w_Xderiv[-1] * V_G1_W
        # J_mid_c = rho_G1_c_Xderiv[-1] * V_G1_C
        # J_in_w = 2 * J_out_w
        # J_in_c = 2 * J_out_c
        # # fluxes
        
        #updating WT in G1
        rho_G1_W[dt_c, 1:] = rho_G1_W[dt_c-1, 1:]  + dt * (-V_G1_W * rho_G1_w_Xderiv[1:] - (par.R_0_d + par.R_cd * n_c_tot[dt_c-1]) * rho_G1_W[dt_c-1, 1:])
        #updating C in G1
        rho_G1_C[dt_c, 1:] = rho_G1_C[dt_c-1, 1:]  + dt * (-V_G1_C * rho_G1_c_Xderiv[1:] )
        #updating WT in S
        rho_S_W[dt_c, 1:]  = rho_S_W[dt_c-1, 1:]  + dt * (-par.V_S * rho_S_w_Xderiv[1:])
        #updating C in S
        rho_S_C[dt_c, 1:]  = rho_S_C[dt_c-1, 1:]  + dt * (-par.V_S * rho_S_c_Xderiv[1:])
        
        
        #updating diff WT
        n_w_d[dt_c]  =  n_w_d[dt_c-1] + dt * ( (par.R_0_d + par.R_cd * n_c_tot[dt_c-1]) * n_w_g1[dt_c-1] - par.R_a * n_w_d[dt_c-1] * n_c_tot[dt_c-1] )
        
        #updating apop WT
        n_w_apop[dt_c] = n_w_apop[dt_c-1] + dt * par.R_a * n_w_d[dt_c-1] * n_c_tot[dt_c-1]
        
        
        # updating collective populations
        n_w_g1[dt_c] = (np.sum(rho_G1_W[dt_c,1:-1]) + 0.5 * rho_G1_W[dt_c,0]+ 0.5 * rho_G1_W[dt_c,-1] )*sol.dx_G1
        n_w_s[dt_c]  = (np.sum(rho_S_W[dt_c,1:-1])  + 0.5 * rho_S_W[dt_c,0]+  0.5 * rho_S_W[dt_c,-1]  )*sol.dx_S
        n_w_tot[dt_c] = n_w_g1[dt_c] + n_w_s[dt_c] + n_w_d[dt_c]
        
        n_c_g1[dt_c] = (np.sum(rho_G1_C[dt_c,1:-1]) + 0.5 * rho_G1_C[dt_c,0]+ 0.5 * rho_G1_C[dt_c,-1] )*sol.dx_G1
        n_c_s[dt_c] =  (np.sum(rho_S_C[dt_c,1:-1])  + 0.5 * rho_S_C[dt_c,0]+  0.5 * rho_S_C[dt_c,-1]  )*sol.dx_S
        n_c_tot[dt_c] =n_c_g1[dt_c] + n_c_s[dt_c]
        
        N_tot[dt_c] = n_w_tot[dt_c] + n_c_tot[dt_c]
        # updating collective populations
        
        # updating M
        # M[dt_c] = M[dt_c-1] + dt * ( par.S_W * n_w_tot[dt_c-1] + par.S_C * n_c_tot[dt_c-1] - n_w_tot[dt_c-1] * concen[dt_c-1] / (concen[dt_c-1] + par.K_W) - n_c_tot[dt_c-1] * concen[dt_c-1] / (concen[dt_c-1] + par.K_C) )
        # concen[dt_c] = M[dt_c]/N_tot[dt_c]
        M[dt_c] = M[dt_c-1] + dt * ( par.S_W * n_w_tot[dt_c-1] + par.S_C * n_c_tot[dt_c-1] - n_w_tot[dt_c-1] * par.MU_W * concen[dt_c-1] - n_c_tot[dt_c-1] *  par.MU_C * concen[dt_c-1] )
        # M[dt_c] = M[dt_c-1] + dt * ( par.S_W * n_w_tot[dt_c] + par.S_C * n_c_tot[dt_c] - n_w_tot[dt_c] * par.MU_W * concen[dt_c-1] - n_c_tot[dt_c] *  par.MU_C * concen[dt_c-1] )
        concen[dt_c] = M[dt_c]/N_tot[dt_c]
        # concen[dt_c] = concen[dt_c-1]
        # updating M
        
        
        
        
        # avg_conc_w = 0.5 * (concen[dt_c-1] + concen[dt_c])
        # avg_conc_c = avg_conc_w
            
        # # V_G1_W = par.F_w * avg_conc_w / (avg_conc_w + par.K_W)
        # # V_G1_C = par.F_c * avg_conc_c / (avg_conc_c + par.K_C)
        
        # V_G1_W =  par.MU_W * avg_conc_w
        # V_G1_C =  par.V_0 + par.MU_C * avg_conc_c
        
        # # I write all x derivatves backward because the flow is to the right
        
        # # these ones update with continuity eq (BC)
        # # if dt_c==1:
        # #     rho_G1_W[dt_c, 0] = 2 * (par.V_S / V_G1_W) * rho_S_W[dt_c-1, -1]
        # #     rho_G1_C[dt_c, 0] = 2 * (par.V_S / V_G1_C) * rho_S_C[dt_c-1, -1]
        # #     rho_S_W[dt_c, 0] =  (V_G1_W / par.V_S) * rho_G1_W[dt_c-1, -1]
        # #     rho_S_C[dt_c, 0] =  (V_G1_C / par.V_S) * rho_G1_C[dt_c-1, -1]
        # # else:
        # #     rho_G1_W[dt_c, 0] = rho_G1_W[dt_c-1, 0] + 2 * (par.V_S / V_G1_W) * (rho_S_W[dt_c-1, -1]-rho_S_W[dt_c-2, -1])
        # #     rho_G1_C[dt_c, 0] = rho_G1_C[dt_c-1, 0] + 2 * (par.V_S / V_G1_C) * (rho_S_C[dt_c-1, -1]-rho_S_C[dt_c-2, -1])
        # #     rho_S_W[dt_c, 0] =  rho_S_W[dt_c-1, 0] + (V_G1_W / par.V_S) * (rho_G1_W[dt_c-1, -1] - rho_G1_W[dt_c-2, -1])
        # #     rho_S_C[dt_c, 0] =  rho_S_C[dt_c-1, 0] + (V_G1_C / par.V_S) * (rho_G1_C[dt_c-1, -1] - rho_G1_C[dt_c-2, -1])
        
        # rho_G1_W[dt_c, 0] = 2 * (par.V_S / V_G1_W) * rho_S_W[dt_c, -1]
        # rho_G1_C[dt_c, 0] = 2 * (par.V_S / V_G1_C) * rho_S_C[dt_c, -1]
        # rho_S_W[dt_c, 0] =  (V_G1_W / par.V_S) * rho_G1_W[dt_c, -1]
        # rho_S_C[dt_c, 0] =  (V_G1_C / par.V_S) * rho_G1_C[dt_c, -1]
        
        
        # # rho_G1_W[dt_c, 0] = 2 * (par.V_S / V_G1_W) * (rho_S_W[dt_c-1, -1] + 0.5 *(rho_S_W[dt_c-1, -1]-rho_S_W[dt_c-1, -2])  ) + 0.5 * (rho_G1_W[dt_c-1, 1] - rho_G1_W[dt_c-1, 0])
        # # rho_G1_C[dt_c, 0] = 2 * (par.V_S / V_G1_C) * rho_S_C[dt_c-1, -1]
        # # rho_S_W[dt_c, 0] =  (V_G1_W / par.V_S) * rho_G1_W[dt_c-1, -1] 
        # # rho_S_C[dt_c, 0] =  (V_G1_C / par.V_S) * rho_G1_C[dt_c-1, -1]
        # # the others update based on the equations
        
        
        # #Xderiv: WT in G1
        # rho_G1_w_Xderiv[1:] =   (rho_G1_W[dt_c, 1:] - rho_G1_W[dt_c, 0:-1])   /  sol.dx_G1 #backward
        # #Xderiv: C in G1
        # rho_G1_c_Xderiv[1:] =   (rho_G1_C[dt_c, 1:] - rho_G1_C[dt_c, 0:-1])   /  sol.dx_G1 #backward
        # #Xderiv: WT in S
        # rho_S_w_Xderiv[1:] =   (rho_S_W[dt_c, 1:] - rho_S_W[dt_c, 0:-1])      /  sol.dx_S #backwardd
        # #Xderiv: C in S
        # rho_S_c_Xderiv[1:] =   (rho_S_C[dt_c, 1:] - rho_S_C[dt_c, 0:-1])      /  sol.dx_S #backwardd
        
        
        
        # # # fluxes
        # # J_out_w = rho_S_w_Xderiv[-1]  * par.V_S
        # # J_out_c = rho_S_c_Xderiv[-1]  * par.V_S
        # # J_mid_w = rho_G1_w_Xderiv[-1] * V_G1_W
        # # J_mid_c = rho_G1_c_Xderiv[-1] * V_G1_C
        # # J_in_w = 2 * J_out_w
        # # J_in_c = 2 * J_out_c
        # # # fluxes
        
        # #updating WT in G1
        # rho_G1_W[dt_c, 1:] = rho_G1_W[dt_c-1, 1:]  + dt * (-V_G1_W * rho_G1_w_Xderiv[1:] - (par.R_0_d + par.R_cd * n_c_tot[dt_c-1]) * rho_G1_W[dt_c-1, 1:])
        # #updating C in G1
        # rho_G1_C[dt_c, 1:] = rho_G1_C[dt_c-1, 1:]  + dt * (-V_G1_C * rho_G1_c_Xderiv[1:] )
        # #updating WT in S
        # rho_S_W[dt_c, 1:]  = rho_S_W[dt_c-1, 1:]  + dt * (-par.V_S * rho_S_w_Xderiv[1:])
        # #updating C in S
        # rho_S_C[dt_c, 1:]  = rho_S_C[dt_c-1, 1:]  + dt * (-par.V_S * rho_S_c_Xderiv[1:])
        
        
        # #updating diff WT
        # n_w_d[dt_c]  =  n_w_d[dt_c-1] + dt * ( (par.R_0_d + par.R_cd * n_c_tot[dt_c-1]) * n_w_g1[dt_c-1] - par.R_a * n_w_d[dt_c-1] * n_c_tot[dt_c-1] )
        
        # #updating apop WT
        # n_w_apop[dt_c] = n_w_apop[dt_c-1] + dt * par.R_a * n_w_d[dt_c-1] * n_c_tot[dt_c-1]
        
        
        # # updating collective populations
        # n_w_g1[dt_c] = (np.sum(rho_G1_W[dt_c,1:-1]) + 0.5 * rho_G1_W[dt_c,0]+ 0.5 * rho_G1_W[dt_c,-1] )*sol.dx_G1
        # n_w_s[dt_c]  = (np.sum(rho_S_W[dt_c,1:-1])  + 0.5 * rho_S_W[dt_c,0]+  0.5 * rho_S_W[dt_c,-1]  )*sol.dx_S
        # n_w_tot[dt_c] = n_w_g1[dt_c] + n_w_s[dt_c] + n_w_d[dt_c]
        
        # n_c_g1[dt_c] = (np.sum(rho_G1_C[dt_c,1:-1]) + 0.5 * rho_G1_C[dt_c,0]+ 0.5 * rho_G1_C[dt_c,-1] )*sol.dx_G1
        # n_c_s[dt_c] =  (np.sum(rho_S_C[dt_c,1:-1])  + 0.5 * rho_S_C[dt_c,0]+  0.5 * rho_S_C[dt_c,-1]  )*sol.dx_S
        # n_c_tot[dt_c] =n_c_g1[dt_c] + n_c_s[dt_c]
        
        # N_tot[dt_c] = n_w_tot[dt_c] + n_c_tot[dt_c]
        # # updating collective populations
        
        # # updating M
        # # M[dt_c] = M[dt_c-1] + dt * ( par.S_W * n_w_tot[dt_c-1] + par.S_C * n_c_tot[dt_c-1] - n_w_tot[dt_c-1] * concen[dt_c-1] / (concen[dt_c-1] + par.K_W) - n_c_tot[dt_c-1] * concen[dt_c-1] / (concen[dt_c-1] + par.K_C) )
        # # concen[dt_c] = M[dt_c]/N_tot[dt_c]
        # M[dt_c] = M[dt_c-1] + dt * ( par.S_W * n_w_tot[dt_c-1] + par.S_C * n_c_tot[dt_c-1] - n_w_tot[dt_c-1] * par.MU_W * concen[dt_c-1] - n_c_tot[dt_c-1] *  par.MU_C * concen[dt_c-1] )
        # # M[dt_c] = M[dt_c-1] + dt * ( par.S_W * n_w_tot[dt_c] + par.S_C * n_c_tot[dt_c] - n_w_tot[dt_c] * par.MU_W * concen[dt_c-1] - n_c_tot[dt_c] *  par.MU_C * concen[dt_c-1] )
        # concen[dt_c] = M[dt_c]/N_tot[dt_c]
        # # concen[dt_c] = concen[dt_c-1]
        # # updating M
        
        
        
    
    # #test plot
    # # WT
    
    # plt.figure()
    # index_to_plot = 0
    # plot_time = sol.time_tilde[index_to_plot]/par.beta_wt
    # xplot = np.concatenate((sol.x_G1, sol.x_S))
    # yplot = np.concatenate((rho_G1_W[index_to_plot,:], rho_S_W[index_to_plot,:]))
    # plt.plot(xplot, yplot, label='WT'+', t='+str(round(plot_time,2)))
    
    # index_to_plot = int(0.01 * sol.L)
    # plot_time = sol.time_tilde[index_to_plot]/par.beta_wt
    # xplot = np.concatenate((sol.x_G1, sol.x_S))
    # yplot = np.concatenate((rho_G1_W[index_to_plot,:], rho_S_W[index_to_plot,:]))
    # plt.plot(xplot, yplot, label='WT'+', t='+str(round(plot_time,2)))
    
    # index_to_plot = int(0.02 * sol.L)
    # plot_time = sol.time_tilde[index_to_plot]/par.beta_wt
    # xplot = np.concatenate((sol.x_G1, sol.x_S))
    # yplot = np.concatenate((rho_G1_W[index_to_plot,:], rho_S_W[index_to_plot,:]))
    # plt.plot(xplot, yplot, label='WT'+', t='+str(round(plot_time,2)))
    # plt.legend()
    
    # index_to_plot = int(0.06 * sol.L)
    # plot_time = sol.time_tilde[index_to_plot]/par.beta_wt
    # xplot = np.concatenate((sol.x_G1, sol.x_S))
    # yplot = np.concatenate((rho_G1_W[index_to_plot,:], rho_S_W[index_to_plot,:]))
    # plt.plot(xplot, yplot, label='WT'+', t='+str(round(plot_time,2)))
    # plt.legend()
    
    
    # plt.figure()
    # index_to_plot = 0
    # plot_time = sol.time_tilde[index_to_plot]/par.beta_wt
    # xplot = np.concatenate((sol.x_G1, sol.x_S))
    # yplot = np.concatenate((rho_G1_C[index_to_plot,:], rho_S_C[index_to_plot,:]))
    # plt.plot(xplot, yplot, label='C'+', t='+str(round(plot_time,2)))
    
    # index_to_plot = int(0.01 * sol.L)
    # plot_time = sol.time_tilde[index_to_plot]/par.beta_wt
    # xplot = np.concatenate((sol.x_G1, sol.x_S))
    # yplot = np.concatenate((rho_G1_C[index_to_plot,:], rho_S_C[index_to_plot,:]))
    # plt.plot(xplot, yplot, label='C'+', t='+str(round(plot_time,2)))
    
    # index_to_plot = int(0.02 * sol.L)
    # plot_time = sol.time_tilde[index_to_plot]/par.beta_wt
    # xplot = np.concatenate((sol.x_G1, sol.x_S))
    # yplot = np.concatenate((rho_G1_C[index_to_plot,:], rho_S_C[index_to_plot,:]))
    # plt.plot(xplot, yplot, label='C'+', t='+str(round(plot_time,2)))
    
    # index_to_plot = int(0.06 * sol.L)
    # plot_time = sol.time_tilde[index_to_plot]/par.beta_wt
    # xplot = np.concatenate((sol.x_G1, sol.x_S))
    # yplot = np.concatenate((rho_G1_C[index_to_plot,:], rho_S_C[index_to_plot,:]))
    # plt.plot(xplot, yplot, label='C'+', t='+str(round(plot_time,2)))
    # plt.legend()
    
    
    
    
    # plt.figure()
    # index_to_plot = 0
    # xplot = np.concatenate((sol.x_G1, sol.x_S))
    # yplot = np.concatenate((rho_G1_W[index_to_plot,:]/n_w_tot[index_to_plot], rho_S_W[index_to_plot,:]/n_w_tot[index_to_plot]))
    # plt.plot(xplot, yplot, label='WT')
    
    # index_to_plot = int(0.1 * sol.L)
    # xplot = np.concatenate((sol.x_G1, sol.x_S))
    # yplot = np.concatenate((rho_G1_W[index_to_plot,:]/n_w_tot[index_to_plot], rho_S_W[index_to_plot,:]/n_w_tot[index_to_plot]))
    # plt.plot(xplot, yplot, label='WT')
    
    # index_to_plot = int(0.5 * sol.L)
    # xplot = np.concatenate((sol.x_G1, sol.x_S))
    # yplot = np.concatenate((rho_G1_W[index_to_plot,:]/n_w_tot[index_to_plot], rho_S_W[index_to_plot,:]/n_w_tot[index_to_plot]))
    # plt.plot(xplot, yplot, label='WT')
    # plt.legend()
    
    # plt.figure()
    # plt.plot(sol.time_tilde, n_w_tot, label='WT tot')
    # plt.plot(sol.time_tilde, n_w_d  , label='WT diff')
    # plt.plot(sol.time_tilde, n_w_g1, label='WT g1')
    # plt.plot(sol.time_tilde, n_w_s, label='WT s/g2/m')
    # plt.yscale("log")
    # plt.legend()
    
    # plt.figure()
    # # plt.plot(sol.time_tilde, n_w_tot/n_w_tot, label='WT tot ratio')
    # plt.plot(sol.time_tilde, n_w_d/n_w_tot  , label='WT diff ratio')
    # plt.plot(sol.time_tilde, n_w_g1/n_w_tot, label='WT g1 ratio')
    # plt.plot(sol.time_tilde, n_w_s/n_w_tot, label='WT s/g2/m ratio')
    # plt.legend()
    # # WT
    
    # # fhgjfgf
    # # C
    # plt.figure()
    # index_to_plot = 0
    # xplot = np.concatenate((sol.x_G1, sol.x_S))
    # yplot = np.concatenate((rho_G1_C[index_to_plot,:]/n_c_tot[index_to_plot], rho_S_C[index_to_plot,:]/n_c_tot[index_to_plot]))
    # plt.plot(xplot, yplot, label='C')
    
    # index_to_plot = int(0.1 * sol.L)
    # xplot = np.concatenate((sol.x_G1, sol.x_S))
    # yplot = np.concatenate((rho_G1_C[index_to_plot,:]/n_c_tot[index_to_plot], rho_S_C[index_to_plot,:]/n_c_tot[index_to_plot]))
    # plt.plot(xplot, yplot, label='C')
    
    # index_to_plot = int(0.5 * sol.L)
    # xplot = np.concatenate((sol.x_G1, sol.x_S))
    # yplot = np.concatenate((rho_G1_C[index_to_plot,:]/n_c_tot[index_to_plot], rho_S_C[index_to_plot,:]/n_c_tot[index_to_plot]))
    # plt.plot(xplot, yplot, label='C')
    # plt.legend()
    
    
    # plt.figure()
    # plt.plot(sol.time_tilde, n_c_tot, label='C tot')
    # plt.plot(sol.time_tilde, n_c_g1, label='C g1')
    # plt.plot(sol.time_tilde, n_c_s, label='C s/g2/m')
    # plt.yscale("log")
    # plt.legend()
    
    # plt.figure()
    # plt.plot(sol.time_tilde, n_c_g1/n_c_tot, label='C g1 ratio')
    # plt.plot(sol.time_tilde, n_c_s/n_c_tot, label='C s/g2/m ratio')
    # plt.legend()
    # # C
    
    # plt.figure()
    # plt.plot(sol.time_tilde, M, label='total M')
    # plt.legend()
    
    # plt.figure()
    # plt.plot(sol.time_tilde, concen, label='concen')
    # plt.legend()
    # # test plot

    # jhg
    
    single_org_sol = single_org_sol_class()
    
    # single_org_sol.rho_G1_W = copy.copy(rho_G1_W)
    # single_org_sol.rho_S_W = copy.copy(rho_S_W)
    single_org_sol.n_w_tot = copy.copy(n_w_tot)
    single_org_sol.n_w_d = copy.copy(n_w_d)
    single_org_sol.n_w_g1 = copy.copy(n_w_g1)
    single_org_sol.n_w_s = copy.copy(n_w_s)
    
    # single_org_sol.rho_G1_C = copy.copy(rho_G1_C)
    # single_org_sol.rho_S_C = copy.copy(rho_S_C)
    single_org_sol.n_c_tot = copy.copy(n_c_tot)
    single_org_sol.n_c_g1 = copy.copy(n_c_g1)
    single_org_sol.n_c_s = copy.copy(n_c_s)
    
    single_org_sol.N_tot = copy.copy(N_tot)
    single_org_sol.M = copy.copy(M)
    
    single_org_sol.init_conc = N_tot[0]/M[0]
    
    single_org_sol.init_wt_frac = (n_w_tot[0]) / (N_tot[0])
    
    
    closest_arg = np.argmin(np.abs(sol.time_tilde/par.beta_wt - par.t_eval))
    single_org_sol.growth_60_wt = n_w_tot[closest_arg]/n_w_tot[0]
    single_org_sol.growth_60_c  = n_c_tot[closest_arg]/n_c_tot[0]
    
    single_org_sol.rho_G1_0_0_w = rho_G1_0_0_w
    single_org_sol.rho_G1_0_0_c = rho_G1_0_0_c
    
    return single_org_sol


def nondim_all_solver(sol, par):
    
    
    # pure wt
    sol.n_0_w_pureW = 1.0
    sol.n_0_c_pureW = 0.0
    sol.n_0_d_pureW = sol.n_0_w_pureW * par.f_D_W
    sol.M_0_pureW   = par.alpha_w*sol.n_0_w_pureW # nondim
    
    # m = (1+par.R_0_d)*(par.alpha_w+par.K_W)/(par.F_w * par.alpha_w)
    # m = (1+par.R_0_d)/(par.MU_W*par.alpha_w)
    # m_prime = 1/par.V_S
    # A = np.exp(-m*np.pi)*(par.alpha_w*par.F_w)/( (par.alpha_w+par.K_W) * par.V_S)
    denom = (1-np.exp(-par.m*np.pi))/par.m + par.A*(1-np.exp(-par.m_prime*np.pi))/par.m_prime
    
    sol.rho_G1_0_0_w_pureW = par.f_P_W * sol.n_0_w_pureW / denom
    sol.rho_G1_0_0_c_pureW = 0
    
    single_org_sol = single_org_sol_class()
    single_org_sol = rho_solver(par, sol, sol.rho_G1_0_0_w_pureW, sol.rho_G1_0_0_c_pureW, sol.n_0_d_pureW , sol.M_0_pureW)
    
    sol.pure_w_sol = copy.copy(single_org_sol)
    # pure wt
    
    # pure C
    sol.n_0_w_pureC = 0.0
    sol.n_0_c_pureC = 1.0
    sol.n_0_d_pureC = 0.0
    sol.M_0_pureC   = par.alpha_c*sol.n_0_c_pureC # nondim
    
    # q = par.gamma*(par.alpha_c+par.K_C)/(par.F_c * par.alpha_c)
    # q = par.gamma / (par.V_0 + par.MU_C * par.alpha_c)
    # q_prime = par.gamma/par.V_S
    # B = np.exp(-q*np.pi)*(par.alpha_c*par.F_c)/( (par.alpha_c+par.K_C) * par.V_S)
    # B = np.exp(-par.q*np.pi)*(par.V_0 + par.MU_C * par.alpha_c) / par.V_S
    denom = (1-np.exp(-par.q*np.pi))/par.q + par.B*(1-np.exp(-par.q_prime*np.pi))/par.q_prime
    
    sol.rho_G1_0_0_w_pureC = 0.0
    sol.rho_G1_0_0_c_pureC = sol.n_0_c_pureC / denom
    
    single_org_sol = single_org_sol_class()
    single_org_sol = rho_solver(par, sol, sol.rho_G1_0_0_w_pureC, sol.rho_G1_0_0_c_pureC, sol.n_0_d_pureC , sol.M_0_pureC)
    
    sol.pure_c_sol = copy.copy(single_org_sol)
    # pure C
    
    
    # mixed
    mixed_sample_bank = np.loadtxt("mixed_sample_bank.csv", delimiter=',', dtype=int)
    size = np.shape(mixed_sample_bank)[0]
    # sample_indices = np.random.randint(0,size,par.n_org)
    # sample_indices = 3*np.array(range(1,par.n_org + 1 ,1))
    
    sample_indices = sample_indices_mix.copy()
    
    n_0_w_list = []
    n_0_c_list = []
    for sample_c in range(par.n_org):
        n_0_w_list.append(mixed_sample_bank[sample_indices[sample_c], 0])
        n_0_c_list.append(mixed_sample_bank[sample_indices[sample_c], 1])
        
        # ratio_big = float(np.random.randint(1,6,1))
        # ratio = np.random.choice([ratio_big, 1/ratio_big])
        # n_0_c_list.append(mixed_sample_bank[sample_indices[sample_c], 0] * ratio)
        
    n_0_w_list = np.array(n_0_w_list)
    n_0_c_list = np.array(n_0_c_list)
    n_0_d_list = n_0_w_list * par.f_D_W
    
    # n_0_w_list = np.random.randint(10, 100, n_org)
    # n_0_c_list = np.random.randint(10, 100, n_org)
    
    # samples = np.random.normal((1-par.frac), par.frac_std, 3*len(n_0_w_list))
    # filtered_samples = np.array(samples[np.abs(samples - (1-par.frac)) <= 2 * par.frac_std])
    # n_0_d_list = n_0_w_list * filtered_samples[:len(n_0_w_list)]
    
    # n_0_d_list = n_0_w_list * par.f_D_W
    # n_0_p_list = n_0_w_list - n_0_d_list
    
    sol.mixed_sol = []
     
    
    # init_wt_frac = n_0_w_list / (n_0_w_list+n_0_c_list)
    # sol.init_wt_frac = init_wt_frac.copy()
    
  
    sol.growth_60_wt = np.zeros(par.n_org)
    sol.growth_60_c  = np.zeros(par.n_org)
    
    # sol.n_w_mix_matrix = np.zeros((par.n_org, sol.L))
    # sol.n_p_mix_matrix = np.zeros((par.n_org, sol.L))
    # sol.n_d_mix_matrix = np.zeros((par.n_org, sol.L))
    # sol.n_c_mix_matrix = np.zeros((par.n_org, sol.L))

    # sol.n_w_norm_mix_matrix = np.zeros((par.n_org, sol.L))
    # sol.n_p_norm_mix_matrix = np.zeros((par.n_org, sol.L))
    # sol.n_d_norm_mix_matrix = np.zeros((par.n_org, sol.L))
    # sol.n_c_norm_mix_matrix = np.zeros((par.n_org, sol.L))
    
    
    for org_c in range(par.n_org):
        
        n_0_w = n_0_w_list[org_c]
        n_0_d = n_0_d_list[org_c]
        n_0_c = n_0_c_list[org_c]
        
        
        alpha_mix = (n_0_w*par.alpha_w+n_0_c*par.alpha_c)/(n_0_w + n_0_c)
        # alpha_mix = np.random.uniform(par.alpha_c, par.alpha_w)
        # alpha_mix = np.random.uniform(0, 10*par.alpha_w)
        # alpha_mix = 3 * par.alpha_w
        # alpha_mix = np.random.uniform(0,  par.alpha_w)
        # alpha_mix = alpha_w
        # alpha_mix = par.alpha_c
        M_0_mix = alpha_mix * (n_0_w + n_0_c)
        
        
        rho_G1_0_0_w = (sol.rho_G1_0_0_w_pureW / sol.n_0_w_pureW) * n_0_w
        rho_G1_0_0_c = (sol.rho_G1_0_0_c_pureC / sol.n_0_c_pureC) * n_0_c
        
        single_org_sol = single_org_sol_class()
        single_org_sol = rho_solver(par, sol, rho_G1_0_0_w, rho_G1_0_0_c, n_0_d , M_0_mix)
        
        sol.mixed_sol.append(copy.copy(single_org_sol))
    #mix
  
    
    return sol

def nondim_temporal_plot_func(time_tilde, n_w_p, n_c_p, n_w_mix_mat, n_c_mix_mat):
    
    plt.figure()
    plt.plot(time_tilde, n_w_p, label='pure WT')
    plt.plot(time_tilde, n_c_p, label='pure C')
    
    n_org = np.shape(n_w_mix_mat)[0]
    
    y_avg = np.mean(n_w_mix_mat, axis=0)
    y_err = np.std(n_w_mix_mat, axis=0)/np.sqrt(n_org)
    plt.plot(time_tilde, y_avg)
    plt.errorbar(time_tilde, y_avg, yerr=y_err, capsize=0, label='mixed WT')
    
    y_avg = np.mean(n_c_mix_mat, axis=0)
    y_err = np.std(n_c_mix_mat, axis=0)/np.sqrt(n_org)
    plt.plot(time_tilde, y_avg)
    plt.errorbar(time_tilde, y_avg, yerr=y_err, capsize=0, label='mixed C')
    
    plt.legend()
    plt.yscale("log")
    
    return 1

def dim_temporal_plot_func(time, n_w_p, n_c_p, n_w_mix_mat, n_c_mix_mat):
    
    plt.figure()
    
    
    C_mix_exp = np.loadtxt("exp_data"+"/"+"C_bar_mix_overal.csv", delimiter=',')
    WT_mix_exp = np.loadtxt("exp_data"+"/"+"WT_bar_mix_overal.csv", delimiter=',')
    C_pure_exp = np.loadtxt("exp_data"+"/"+"overal_C_pure.csv", delimiter=',')
    WT_pure_exp = np.loadtxt("exp_data"+"/"+"overal_WT_pure.csv", delimiter=',')
    
    
    #C_mix
    # plt.scatter(C_mix_exp[0,:], C_mix_exp[1,:], label='exp_C_mix')
    plt.errorbar(C_mix_exp[0,:], C_mix_exp[1,:], yerr=C_mix_exp[2,:], fmt='s', color='g', ecolor='r', capsize=2, label='mixed C (exp)', markerfacecolor='none')
    
    #WT_mix
    # plt.scatter(WT_mix_exp[0,:], WT_mix_exp[1,:], label='exp_WT_mix')
    plt.errorbar(WT_mix_exp[0,:], WT_mix_exp[1,:], yerr=WT_mix_exp[2,:], fmt='s', color='m', ecolor='r', capsize=2, label='mixed WT (exp)', markerfacecolor='none')
    
    #C_pure
    # plt.scatter(C_pure_exp[0,:], C_pure_exp[1,:], label='exp_C_pure')
    plt.errorbar(C_pure_exp[0,:], C_pure_exp[1,:], yerr=C_pure_exp[2,:], fmt='o', color='g', ecolor='r', capsize=2, label='pure C (exp)')
    
    #WT_pure
    # plt.scatter(WT_pure_exp[0,:], WT_pure_exp[1,:], label='exp_WT_pure')
    plt.errorbar(WT_pure_exp[0,:], WT_pure_exp[1,:], yerr=WT_pure_exp[2,:], fmt='o', color='m', ecolor='r', capsize=2, label='pure WT (exp)')
        
    
        
    plt.plot(time, n_w_p, label='pure WT (model)', color = 'm', linestyle='--')
    plt.plot(time, n_c_p, label='pure C (model)', color = 'g', linestyle='--')
    
    n_org = np.shape(n_w_mix_mat)[0]
    
    y_avg = np.mean(n_w_mix_mat, axis=0)
    y_err = np.std(n_w_mix_mat, axis=0)/np.sqrt(n_org)
    # plt.plot(time, y_avg, color = 'm', linestyle='--')
    plt.errorbar(time, y_avg, yerr=y_err, capsize=0, label='mixed WT (model)', color = 'm')
    
    y_avg = np.mean(n_c_mix_mat, axis=0)
    y_err = np.std(n_c_mix_mat, axis=0)/np.sqrt(n_org)
    # plt.plot(time, y_avg)
    plt.errorbar(time, y_avg, yerr=y_err, capsize=0, label='mixed C (model)', color = 'g')
    
    plt.xlabel('time(h)', fontsize=15)
    plt.ylabel(r'$\overline{N(t)/N(0)}$', fontsize=15)
    # plt.grid()
    plt.xticks( fontsize=15)
    plt.yticks(fontsize=15)
    plt.yscale("log")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig('mean_field_four_pops.PNG', dpi=300)
    
    # plt.figure()
    # WT_mix_model =  np.log(np.mean(n_w_mix_mat, axis=0))
    # diff = (WT_mix_model[1:]- WT_mix_model[:-1]) / ( time[1:]- time[:-1] )
    # plt.plot(time[:-1],diff)
    
    return 1

def compos_effect_plotter(sol, par):
    
    n_org = par.n_org
    
    init_wt_frac = np.zeros(n_org)
    growth_60_wt = np.zeros(n_org)
    growth_60_c  = np.zeros(n_org)
    
    cmap = 'Greys'
    # cmap = 'viridis'
    
    n_w_mix_mat_abs = np.zeros((n_org, sol.L))
    n_c_mix_mat_abs = np.zeros((n_org, sol.L))
    
    n_w_mix_mat = np.zeros((n_org, sol.L))
    n_c_mix_mat = np.zeros((n_org, sol.L))
    for org_c in range(n_org):
        n_w_mix_mat[org_c,:] = sol.mixed_sol[org_c].n_w_tot / sol.mixed_sol[org_c].n_w_tot[0]
        n_c_mix_mat[org_c,:] = sol.mixed_sol[org_c].n_c_tot / sol.mixed_sol[org_c].n_c_tot[0]
        
        n_w_mix_mat_abs[org_c,:] = sol.mixed_sol[org_c].n_w_tot
        n_c_mix_mat_abs[org_c,:] = sol.mixed_sol[org_c].n_c_tot
            
        init_wt_frac[org_c] = sol.mixed_sol[org_c].init_wt_frac
        growth_60_wt[org_c] = sol.mixed_sol[org_c].growth_60_wt
        growth_60_c[org_c] =  sol.mixed_sol[org_c].growth_60_c
        
    init_compos_data = np.loadtxt("exp_data"+"/"+"init_compos_data_exp.csv", delimiter=',')
    
    
    # values_for_color = np.log10(n_w_mix_matrix[:,0] * n_c_mix_matrix[:,0])
    # values_for_color = np.log10((n_w_mix_mat_abs * n_c_mix_mat_abs)[:,0])
    values_for_color = np.ones(n_org)
    
    # values_for_color = n_w_mix_matrix[:,0] * n_c_mix_matrix[:,0] / ((n_w_mix_matrix[:,0] + n_c_mix_matrix[:,0])**2)

    # Use 'c' for color values and 'cmap' for the color mapping

    plt.figure()
    # plt.scatter(init_wt_frac, growth_60_wt, 
    #             c=values_for_color,   # <-- this controls point color by value
    #             cmap=cmap,       # <-- choose a colormap you like
    #             s=5,                 # size of the points
    #             label='model')
    # plt.colorbar(label='Color by this vector')
    plt.scatter(init_wt_frac, growth_60_wt, 
                s=5,                 # size of the points
                color='k',
                label='model')
    
    
    plt.scatter(init_compos_data[0,:], init_compos_data[1,:], color='m', marker='*', label= 'exp', zorder=10, s=50)
    plt.xlabel('initial WT fraction', fontsize=15)
    plt.xticks(fontsize=15)
    # plt.ylabel('WT Normalized Number at t=60h', fontsize=12)
    plt.ylabel('WT(t=60h)/WT(0)', fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig('mean_field_WT_compos.PNG', dpi=300)
    
    
    plt.figure()
    # plt.scatter((1-init_wt_frac), growth_60_c, color='grey', label='model', s=2)
    # plt.scatter((1-init_wt_frac), growth_60_c, 
    #             c=values_for_color,   # <-- this controls point color by value
    #             cmap=cmap,       # <-- choose a colormap you like
    #             s=5,                 # size of the points
    #             label='model')
    # plt.colorbar(label='Color by this vector')
    # plt.scatter((1-init_wt_frac), growth_60_c, color='grey', label='model', s=2)
    plt.scatter((1-init_wt_frac), growth_60_c, 
                s=5,                 # size of the points
                color='k',
                label='model')
    plt.scatter(init_compos_data[2,:], init_compos_data[3,:], color='g', marker='*', label='exp', zorder=10, s=50)
    plt.xlabel('initial C fraction', fontsize=15)
    plt.xticks(fontsize=15)
    # plt.ylabel('C Normalized Number at t=60h', fontsize=12)
    plt.ylabel('C(t=60h)/C(0)', fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig('mean_field_C_compos.PNG', dpi=300)
    
    
    # plt.figure()
    # plt.scatter(values_for_color, growth_60_wt, s=2)
    
    # plt.figure()
    # plt.scatter(values_for_color, growth_60_c, s=2)
    
    return

def composition_vs_time_plotter(sol, par):
    
    n_org = par.n_org
    
    time = sol.time_tilde / par.beta_wt
    
    frac_w_g1_pureW = sol.pure_w_sol.n_w_g1 / sol.pure_w_sol.n_w_tot
    frac_w_s_pureW  = sol.pure_w_sol.n_w_s  / sol.pure_w_sol.n_w_tot
    frac_w_d_pureW  = sol.pure_w_sol.n_w_d  / sol.pure_w_sol.n_w_tot
    
    frac_c_g1_pureC  = sol.pure_c_sol.n_c_g1 / sol.pure_c_sol.n_c_tot
    frac_c_s_pureC   = sol.pure_c_sol.n_c_s  / sol.pure_c_sol.n_c_tot
    
    frac_w_g1_mixed_matrix  = np.zeros((n_org, sol.L))
    frac_w_s_mixed_matrix   = np.zeros((n_org, sol.L))
    frac_w_d_mixed_matrix   = np.zeros((n_org, sol.L))
    
    frac_c_g1_mixed_matrix  = np.zeros((n_org, sol.L))
    frac_c_s_mixed_matrix   = np.zeros((n_org, sol.L))
    
    
    for org_c in range(n_org):
        frac_w_g1_mixed_matrix[org_c,:]  = sol.mixed_sol[org_c].n_w_g1 / sol.mixed_sol[org_c].n_w_tot
        frac_w_s_mixed_matrix[org_c,:]   = sol.mixed_sol[org_c].n_w_s  / sol.mixed_sol[org_c].n_w_tot
        frac_w_d_mixed_matrix[org_c,:]   = sol.mixed_sol[org_c].n_w_d  / sol.mixed_sol[org_c].n_w_tot
        
        frac_c_g1_mixed_matrix[org_c,:]  = sol.mixed_sol[org_c].n_c_g1 / sol.mixed_sol[org_c].n_c_tot
        frac_c_s_mixed_matrix[org_c,:]   = sol.mixed_sol[org_c].n_c_s  / sol.mixed_sol[org_c].n_c_tot
    
    
    # plot WT
    x = time
    plt.figure()
    
    # G1
    y = frac_w_g1_pureW
    plt.plot(x, y, label="WT G1 frac - pure WT", color = 'r')
    
    y = np.mean(frac_w_g1_mixed_matrix, axis=0)
    y_err = np.std(frac_w_g1_mixed_matrix, axis=0)/np.sqrt(n_org)
    y_upper = y + y_err
    y_lower = y - y_err
    plt.plot(x, y, label="WT G1 frac - mixed", color = 'r', linestyle = '--')
    plt.fill_between(x, y_lower, y_upper, alpha=0.2, color = 'r')
    # G1
    
    
    # S
    y = frac_w_s_pureW
    plt.plot(x, y, label="WT S frac - pure WT", color = 'tab:olive')
    
    
    
    y = np.mean(frac_w_s_mixed_matrix, axis=0)
    y_err = np.std(frac_w_s_mixed_matrix, axis=0)/np.sqrt(n_org)
    y_upper = y + y_err
    y_lower = y - y_err
    plt.plot(x, y, label="WT S frac - mixed", color = 'tab:olive', linestyle = '--')
    plt.fill_between(x, y_lower, y_upper, alpha=0.2, color = 'tab:olive')
    
#     plt.errorbar(
#     x=[60], y=[0.5*(0.47+0.21)],
#     yerr=[0.03],
#     fmt='o',          # “o” → filled‑circle marker (bullet)
#     markersize=6,     # make the bullet a bit easier to see
#     color='tab:olive',        # line & marker color
#     ecolor='tab:olive',       # error‑bar color (keep it matching)
#     capsize=4,        # length of the whisker caps (points)
#     capthick=1.3,     # cap line thickness (optional)
#     label='S frac – pure (exp)'
# )
#     plt.errorbar(
#     x=[60], y=[0.5*(0.27+0.1)],
#     yerr=[0.03],
#     fmt='s',          # “o” → filled‑circle marker (bullet)
#     markersize=6,     # make the bullet a bit easier to see
#     color='tab:olive',        # line & marker color
#     ecolor='tab:olive',       # error‑bar color (keep it matching)
#     capsize=4,        # length of the whisker caps (points)
#     capthick=1.3,     # cap line thickness (optional)
#     label='S frac – mixed (exp)'
# )
    # S
    
    # D
    y = frac_w_d_pureW
    plt.plot(x, y, label="WT D frac - pure WT", color = 'grey')
    
    y = np.mean(frac_w_d_mixed_matrix, axis=0)
    y_err = np.std(frac_w_d_mixed_matrix, axis=0)/np.sqrt(n_org)
    y_upper = y + y_err
    y_lower = y - y_err
    plt.plot(x, y, label="WT D frac - mixed", color = 'grey', linestyle='--')
    plt.fill_between(x, y_lower, y_upper, alpha=0.2, color = 'grey')
    # plt.axvline(x=0, ymin=0, ymax=1, linestyle='--', color='grey')
    # plt.axvline(x=60, ymin=0, ymax=1, linestyle='--', color='grey')
#     plt.errorbar(
#     x=[60], y=[0.2],
#     yerr=[0.03],
#     fmt='o',          # “o” → filled‑circle marker (bullet)
#     markersize=6,     # make the bullet a bit easier to see
#     color='grey',        # line & marker color
#     ecolor='grey',       # error‑bar color (keep it matching)
#     capsize=4,        # length of the whisker caps (points)
#     capthick=1.3,     # cap line thickness (optional)
#     label='D frac – pure (exp)'
# )
#     plt.errorbar(
#     x=[60], y=[0.29],
#     yerr=[0.03],
#     fmt='s',          # “o” → filled‑circle marker (bullet)
#     markersize=6,     # make the bullet a bit easier to see
#     color='grey',        # line & marker color
#     ecolor='grey',       # error‑bar color (keep it matching)
#     capsize=4,        # length of the whisker caps (points)
#     capthick=1.3,     # cap line thickness (optional)
#     label='D frac – mixed (exp)'
# )
    # D
    
    
    # plt.errorbar(x=[], y=[], yerr=[], color='m', label='')
    
    
    
#     plt.errorbar(
#     x=[0], y=[0.5*(0.47+0.21)],
#     yerr=[0.03],
#     fmt='o',          # “o” → filled‑circle marker (bullet)
#     markersize=6,     # make the bullet a bit easier to see
#     color='m',        # line & marker color
#     ecolor='m',       # error‑bar color (keep it matching)
#     style = '--',
#     capsize=4,        # length of the whisker caps (points)
#     capthick=1.3,     # cap line thickness (optional)
#     label='WT G1 frac – pure WT (exp)'
# )
    
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.xlabel("time (h)", fontsize=15)
    plt.ylabel("fraction (WT)", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim((0.1, 0.6))
    plt.tight_layout()
    plt.savefig("WT_fracrions.PNG", dpi=400)
    # plot WT
    
    # plot C
    x = time
    plt.figure()
    
    #G1
    y = frac_c_g1_pureC
    plt.plot(x, y, label="C G1 frac - pure C", color = 'r')
    
    y = np.mean(frac_c_g1_mixed_matrix, axis=0)
    y_err = np.std(frac_c_g1_mixed_matrix, axis=0)/np.sqrt(n_org)
    y_upper = y + y_err
    y_lower = y - y_err
    plt.plot(x, y, label="C G1 frac - mixed", color = 'r', linestyle = '--')
    plt.fill_between(x, y_lower, y_upper, alpha=0.2, color = 'r')
    #G1
    
    # S
    y = frac_c_s_pureC
    plt.plot(x, y, label="C S frac - pure C", color = 'tab:olive')
    
    y = np.mean(frac_c_s_mixed_matrix, axis=0)
    y_err = np.std(frac_c_s_mixed_matrix, axis=0)/np.sqrt(n_org)
    y_upper = y + y_err
    y_lower = y - y_err
    plt.plot(x, y, label="C S frac - mixed", color = 'tab:olive', linestyle='--')
    plt.fill_between(x, y_lower, y_upper, alpha=0.2, color = 'tab:olive')
    # S
    
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.xlabel("time (h)", fontsize=15)
    plt.ylabel("fraction (C)", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim((0.2, 0.8))
    plt.tight_layout()
    plt.savefig("C_fracrions.PNG", dpi=400)
    # plot C
    
        
    return

def model_matrix_maker(sol_dum, par_dum):
    
    n_org = par_dum.n_org
    
    L = sol_dum.L
    time_tilde = sol_dum.time_tilde
    time = time_tilde / par_dum.beta_wt
    n_w_p = sol_dum.pure_w_sol.n_w_tot
    n_c_p = sol_dum.pure_c_sol.n_c_tot
    
    
    n_w_mix_matrix = np.zeros((n_org, sol_dum.L))
    n_c_mix_matrix = np.zeros((n_org, sol_dum.L))
    
    for org_c in range(n_org):
        n_w_mix_matrix[org_c,:] = sol_dum.mixed_sol[org_c].n_w_tot / sol_dum.mixed_sol[org_c].n_w_tot[0]
        n_c_mix_matrix[org_c,:] = sol_dum.mixed_sol[org_c].n_c_tot / sol_dum.mixed_sol[org_c].n_c_tot[0]
    
    C_mix_model   = np.zeros((3,L))
    WT_mix_model  = np.zeros((3,L))
    C_pure_model  = np.zeros((3,L))
    WT_pure_model = np.zeros((3,L))

    C_mix_model[0,:]   = time
    WT_mix_model[0,:]  = time
    C_pure_model[0,:]  = time
    WT_pure_model[0,:] = time

    C_mix_model[1,:] = np.mean(n_c_mix_matrix, axis=0)
    WT_mix_model[1,:] = np.mean(n_w_mix_matrix, axis=0)
    C_pure_model[1,:] = (n_c_p/n_c_p[0])
    WT_pure_model[1,:] = (n_w_p/n_w_p[0])

    C_mix_model[2,:] = np.std(n_c_mix_matrix, axis=0)/np.sqrt(n_org)
    WT_mix_model[2,:] = np.std(n_w_mix_matrix, axis=0)/np.sqrt(n_org)
    C_pure_model[2,:] = 0.0 * C_pure_model[2,:]
    WT_pure_model[2,:] = 0.0 * WT_pure_model[2,:]
    
    return WT_pure_model, C_pure_model, WT_mix_model, C_mix_model

def total_cost_calc(WT_pure_exp, WT_pure_model, C_pure_exp, C_pure_model, WT_mix_exp, WT_mix_model, C_mix_exp, C_mix_model, sol_dum, par_dum):
    
    cost = dict()
    
    # cost pop
    cost['pop'] = 0.0
    # cost['pop'] +=  0.05*cost_calc_detail(WT_pure_exp, WT_pure_model)
    # cost['pop'] +=  0.05*cost_calc_detail(C_pure_exp, C_pure_model)
    cost_WT_mix, unwighted_list_WT_mix = cost_calc_detail(WT_mix_exp, WT_mix_model)
    cost_C_mix,  unwighted_list_C_mix  = cost_calc_detail(C_mix_exp, C_mix_model)
    # cost['pop'] +=  0.45*cost_calc_detail(WT_mix_exp, WT_mix_model)
    # cost['pop'] +=  0.45*cost_calc_detail(C_mix_exp, C_mix_model)
    cost['pop'] = 0.5 * cost_WT_mix + 0.5 * cost_C_mix
    # scale_pop = np.mean(np.concatenate((unwighted_list_WT_mix, unwighted_list_C_mix)))
    scale_pop = 0.5 * np.mean((WT_mix_exp[2,:]/WT_mix_exp[1,:])**2) + 0.5 * np.mean((C_mix_exp[2,:]/C_mix_exp[1,:])**2)
    # cost pop
    
    # cost T_d
    g_lower_w = 0.3
    g_upper_w = 0.5
    # slope = -2.5 
    # T_g_lower_w = 20.5 + slope * g_lower_w
    # T_g_upper_w = 20.5 + slope * g_upper_w
    
    g_calculated = par_dum.g_calc
    
    # if g_calculated<g_lower_w:
    #     cost['g'] = 500*(g_calculated-g_lower_w)**2
    # elif g_calculated>g_upper_w:
    #     cost['g']= 500*(g_calculated-g_upper_w)**2
    # else:
    #     cost['g']= 0
    cost['g']= 0
    # cost T_d
    
    # percentages
    n_org = par_dum.n_org
    
    time = sol_dum.time_tilde / par_dum.beta_wt
    
    frac_w_g1_pureW = sol_dum.pure_w_sol.n_w_g1 / sol_dum.pure_w_sol.n_w_tot
    frac_w_s_pureW  = sol_dum.pure_w_sol.n_w_s  / sol_dum.pure_w_sol.n_w_tot
    frac_w_d_pureW  = sol_dum.pure_w_sol.n_w_d  / sol_dum.pure_w_sol.n_w_tot
    
    frac_w_g1_mixed_matrix  = np.zeros((n_org, sol_dum.L))
    frac_w_s_mixed_matrix   = np.zeros((n_org, sol_dum.L))
    frac_w_d_mixed_matrix   = np.zeros((n_org, sol_dum.L))
    
    for org_c in range(n_org):
        frac_w_g1_mixed_matrix[org_c,:]  = sol_dum.mixed_sol[org_c].n_w_g1 / sol_dum.mixed_sol[org_c].n_w_tot
        frac_w_s_mixed_matrix[org_c,:]   = sol_dum.mixed_sol[org_c].n_w_s  / sol_dum.mixed_sol[org_c].n_w_tot
        frac_w_d_mixed_matrix[org_c,:]   = sol_dum.mixed_sol[org_c].n_w_d  / sol_dum.mixed_sol[org_c].n_w_tot

        
    eval_index =  np.argmin(np.abs(time - par_dum.t_eval))
    avg_d_perc = np.mean(frac_w_d_mixed_matrix, axis=0)
    avg_s_perc = np.mean(frac_w_s_mixed_matrix, axis=0)
    
    # cost['stat']= 0.5*(avg_d_perc[eval_index]-1.5*frac_w_d_pureW[eval_index])**2 + 0.5*(avg_s_perc[eval_index]-0.5 * frac_w_s_pureW[eval_index])**2
    scale_stat = 0.03**2
    
    cost['stat']= ( 0.333*(avg_d_perc[eval_index]-1.45*frac_w_d_pureW[eval_index])**2 + 0.333*(avg_s_perc[eval_index]-0.57 * frac_w_s_pureW[eval_index])**2 + 0.333 * (frac_w_s_pureW[0]- 0.5*(0.21 +0.47))**2 ) * scale_pop / scale_stat
    
    # percentages
    
    #compos
    n_org = par.n_org
    
    init_wt_frac = np.zeros(n_org)
    growth_60_wt = np.zeros(n_org)
    growth_60_c  = np.zeros(n_org)
    
    cmap = 'Greys'
    # cmap = 'viridis'
    
    n_w_mix_mat_abs = np.zeros((n_org, sol.L))
    n_c_mix_mat_abs = np.zeros((n_org, sol.L))
    
    n_w_mix_mat = np.zeros((n_org, sol.L))
    n_c_mix_mat = np.zeros((n_org, sol.L))
    for org_c in range(n_org):
        n_w_mix_mat[org_c,:] = sol.mixed_sol[org_c].n_w_tot / sol.mixed_sol[org_c].n_w_tot[0]
        n_c_mix_mat[org_c,:] = sol.mixed_sol[org_c].n_c_tot / sol.mixed_sol[org_c].n_c_tot[0]
        
        n_w_mix_mat_abs[org_c,:] = sol.mixed_sol[org_c].n_w_tot
        n_c_mix_mat_abs[org_c,:] = sol.mixed_sol[org_c].n_c_tot
            
        init_wt_frac[org_c] = sol.mixed_sol[org_c].init_wt_frac
        growth_60_wt[org_c] = sol.mixed_sol[org_c].growth_60_wt
        growth_60_c[org_c] =  sol.mixed_sol[org_c].growth_60_c
        
    init_compos_data = np.loadtxt("exp_data"+"/"+"init_compos_data_exp.csv", delimiter=',')
    
    init_c_frac = 1- init_wt_frac
    w_log_compos_fit_exp = np.polyfit(init_compos_data[0,:], np.log(init_compos_data[1,:]), 1)
    c_log_compos_fit_exp = np.polyfit(init_compos_data[2,:], np.log(init_compos_data[3,:]), 1)
    
    w_compos_err_list = (np.log(growth_60_wt) - (w_log_compos_fit_exp[0]*init_wt_frac + w_log_compos_fit_exp[1]) )**2
    c_compos_err_list = (np.log(growth_60_c)  - (c_log_compos_fit_exp[0]*init_c_frac  + c_log_compos_fit_exp[1]) )**2
    
    scale_compos_w = np.var(np.log(init_compos_data[1,:])  - (w_log_compos_fit_exp[0]*init_compos_data[0,:] + w_log_compos_fit_exp[1]) )
    scale_compos_c = np.var(np.log(init_compos_data[3,:])  - (c_log_compos_fit_exp[0]*init_compos_data[2,:] + c_log_compos_fit_exp[1]) )
    
    # scale_compos_w = scale_pop/np.var(np.log(init_compos_data[1,:]))
    # scale_compos_c = scale_pop/np.var(np.log(init_compos_data[3,:]))
    # scale_compos_c = scale_pop/scale_compos_w
    # compos_scale_c = scale_pop/scale_compos_c
    
    # cost['compos'] = 0.5*scale_compos_w*np.mean(w_compos_err_list) + 0.5*scale_compos_c*np.mean(c_compos_err_list)
    
    cost['compos'] = (0.5*np.mean(w_compos_err_list)/scale_compos_w + 0.5*np.mean(c_compos_err_list)/scale_compos_c) * scale_pop
    # fit_slope_w_model = np.polyfit(init_wt_frac, np.log(growth_60_wt), 1)[0]
    # fit_slope_c_model = np.polyfit(1-init_wt_frac, np.log(growth_60_c), 1)[0]
    # fit_slope_w_exp =   np.polyfit(init_compos_data[0,:], np.log(init_compos_data[1,:]), 1)[0]
    # fit_slope_c_exp =   np.polyfit(init_compos_data[2,:], np.log(init_compos_data[3,:]), 1)[0]
    # cost['compos'] = compos_scale*(0.5*((fit_slope_w_model - fit_slope_w_exp)/fit_slope_w_exp)**2 + ((fit_slope_c_model - fit_slope_c_exp)/fit_slope_c_exp)**2)
    #compos
    
    # cost_tot = 0.4 * cost['pop'] + 0.2* cost['g'] + 0.2 * cost['stat'] + 0.2 * cost['compos']
    cost_tot = 0.333 * cost['pop'] + 0.0* cost['g'] + 0.333 * cost['stat'] + 0.333 * cost['compos']
    
    return cost_tot

def grad_evaluator(par_dum, sol_dum, gd):
    
    grad = dict()
    
    
    # # K_W
    # par_plus  = params_class()
    # par_plus  = copy.copy(par_dum)
    # par_plus.K_W = par_dum.K_W + gd.d_K_W
    # par_plus = par_calculator(par_plus)
    # sol_plus  = solution_class()
    # sol_plus  = sol_obj_maker(sol_plus, par_plus)
    # sol_plus = nondim_all_solver(sol_plus, par_plus)
    # WT_pure_model, C_pure_model, WT_mix_model, C_mix_model = model_matrix_maker(sol_plus, par_plus)
    # cost_plus = total_cost_calc(WT_pure_exp, WT_pure_model, C_pure_exp, C_pure_model, WT_mix_exp, WT_mix_model, C_mix_exp, C_mix_model)
    
    # par_minus  = params_class()
    # par_minus  = copy.copy(par_dum)
    # par_minus.K_W = par_dum.K_W - gd.d_K_W
    # par_minus = par_calculator(par_minus)
    # sol_minus  = solution_class()
    # sol_minus  = sol_obj_maker(sol_minus, par_minus)
    # sol_minus = nondim_all_solver(sol_minus, par_minus)
    # WT_pure_model, C_pure_model, WT_mix_model, C_mix_model = model_matrix_maker(sol_minus, par_minus)
    # cost_minus = total_cost_calc(WT_pure_exp, WT_pure_model, C_pure_exp, C_pure_model, WT_mix_exp, WT_mix_model, C_mix_exp, C_mix_model)
    
    # grad['K_W'] = (cost_plus - cost_minus) / (2 * gd.d_K_W)
    # # K_W
    
    # # K_C
    # par_plus  = params_class()
    # par_plus  = copy.copy(par_dum)
    # par_plus.K_C = par_dum.K_C + gd.d_K_C
    # par_plus = par_calculator(par_plus)
    # sol_plus  = solution_class()
    # sol_plus  = sol_obj_maker(sol_plus, par_plus)
    # sol_plus = nondim_all_solver(sol_plus, par_plus)
    # WT_pure_model, C_pure_model, WT_mix_model, C_mix_model = model_matrix_maker(sol_plus, par_plus)
    # cost_plus = total_cost_calc(WT_pure_exp, WT_pure_model, C_pure_exp, C_pure_model, WT_mix_exp, WT_mix_model, C_mix_exp, C_mix_model)
    
    # par_minus  = params_class()
    # par_minus  = copy.copy(par_dum)
    # par_minus.K_C = par_dum.K_C - gd.d_K_C
    # par_minus = par_calculator(par_minus)
    # sol_minus  = solution_class()
    # sol_minus  = sol_obj_maker(sol_minus, par_minus)
    # sol_minus = nondim_all_solver(sol_minus, par_minus)
    # WT_pure_model, C_pure_model, WT_mix_model, C_mix_model = model_matrix_maker(sol_minus, par_minus)
    # cost_minus = total_cost_calc(WT_pure_exp, WT_pure_model, C_pure_exp, C_pure_model, WT_mix_exp, WT_mix_model, C_mix_exp, C_mix_model)
    
    # grad['K_C'] = (cost_plus - cost_minus) / (2 * gd.d_K_C)
    # # K_C
    
    # # S
    # par_plus  = params_class()
    # par_plus  = copy.copy(par_dum)
    # par_plus.S = par_dum.S + gd.d_S
    # par_plus = par_calculator(par_plus)
    # sol_plus  = solution_class()
    # sol_plus  = sol_obj_maker(sol_plus, par_plus)
    # sol_plus = nondim_all_solver(sol_plus, par_plus)
    # WT_pure_model, C_pure_model, WT_mix_model, C_mix_model = model_matrix_maker(sol_plus, par_plus)
    # cost_plus = total_cost_calc(WT_pure_exp, WT_pure_model, C_pure_exp, C_pure_model, WT_mix_exp, WT_mix_model, C_mix_exp, C_mix_model)
    
    # par_minus  = params_class()
    # par_minus  = copy.copy(par_dum)
    # par_minus.S = par_dum.S - gd.d_S
    # par_minus = par_calculator(par_minus)
    # sol_minus  = solution_class()
    # sol_minus  = sol_obj_maker(sol_minus, par_minus)
    # sol_minus = nondim_all_solver(sol_minus, par_minus)
    # WT_pure_model, C_pure_model, WT_mix_model, C_mix_model = model_matrix_maker(sol_minus, par_minus)
    # cost_minus = total_cost_calc(WT_pure_exp, WT_pure_model, C_pure_exp, C_pure_model, WT_mix_exp, WT_mix_model, C_mix_exp, C_mix_model)
    
    # grad['S'] = (cost_plus - cost_minus) / (2 * gd.d_S)
    # # S
    
    # S
    par_plus  = params_class()
    par_plus  = copy.copy(par_dum)
    par_plus.S_W = par_dum.S_W + gd.d_S
    par_plus.S_C = par_dum.S_C + gd.d_S
    par_plus = par_calculator(par_plus)
    sol_plus  = solution_class()
    sol_plus  = sol_obj_maker(sol_plus, par_plus)
    sol_plus = nondim_all_solver(sol_plus, par_plus)
    WT_pure_model, C_pure_model, WT_mix_model, C_mix_model = model_matrix_maker(sol_plus, par_plus)
    cost_plus = total_cost_calc(WT_pure_exp, WT_pure_model, C_pure_exp, C_pure_model, WT_mix_exp, WT_mix_model, C_mix_exp, C_mix_model, sol_plus, par_plus)
    
    par_minus  = params_class()
    par_minus  = copy.copy(par_dum)
    par_minus.S_W = par_dum.S_W - gd.d_S
    par_minus.S_C = par_dum.S_C - gd.d_S
    par_minus = par_calculator(par_minus)
    sol_minus  = solution_class()
    sol_minus  = sol_obj_maker(sol_minus, par_minus)
    sol_minus = nondim_all_solver(sol_minus, par_minus)
    WT_pure_model, C_pure_model, WT_mix_model, C_mix_model = model_matrix_maker(sol_minus, par_minus)
    cost_minus = total_cost_calc(WT_pure_exp, WT_pure_model, C_pure_exp, C_pure_model, WT_mix_exp, WT_mix_model, C_mix_exp, C_mix_model, sol_minus, par_minus)
    
    grad['S'] = (cost_plus - cost_minus) / (2 * gd.d_S)
    print("Grad S : "+str(grad['S']))
    # S
    
    
    # # S_W
    # par_plus  = params_class()
    # par_plus  = copy.copy(par_dum)
    # par_plus.S_W = par_dum.S_W + gd.d_S_W
    # par_plus = par_calculator(par_plus)
    # sol_plus  = solution_class()
    # sol_plus  = sol_obj_maker(sol_plus, par_plus)
    # sol_plus = nondim_all_solver(sol_plus, par_plus)
    # WT_pure_model, C_pure_model, WT_mix_model, C_mix_model = model_matrix_maker(sol_plus, par_plus)
    # cost_plus = total_cost_calc(WT_pure_exp, WT_pure_model, C_pure_exp, C_pure_model, WT_mix_exp, WT_mix_model, C_mix_exp, C_mix_model)
    
    # par_minus  = params_class()
    # par_minus  = copy.copy(par_dum)
    # par_minus.S_W = par_dum.S_W - gd.d_S_W
    # par_minus = par_calculator(par_minus)
    # sol_minus  = solution_class()
    # sol_minus  = sol_obj_maker(sol_minus, par_minus)
    # sol_minus = nondim_all_solver(sol_minus, par_minus)
    # WT_pure_model, C_pure_model, WT_mix_model, C_mix_model = model_matrix_maker(sol_minus, par_minus)
    # cost_minus = total_cost_calc(WT_pure_exp, WT_pure_model, C_pure_exp, C_pure_model, WT_mix_exp, WT_mix_model, C_mix_exp, C_mix_model)
    
    # grad['S_W'] = (cost_plus - cost_minus) / (2 * gd.d_S_W)
    # # S_W
    
    # # S_C
    # par_plus  = params_class()
    # par_plus  = copy.copy(par_dum)
    # par_plus.S_C = par_dum.K_C + gd.d_S_C
    # par_plus = par_calculator(par_plus)
    # sol_plus  = solution_class()
    # sol_plus  = sol_obj_maker(sol_plus, par_plus)
    # sol_plus = nondim_all_solver(sol_plus, par_plus)
    # WT_pure_model, C_pure_model, WT_mix_model, C_mix_model = model_matrix_maker(sol_plus, par_plus)
    # cost_plus = total_cost_calc(WT_pure_exp, WT_pure_model, C_pure_exp, C_pure_model, WT_mix_exp, WT_mix_model, C_mix_exp, C_mix_model)
    
    # par_minus  = params_class()
    # par_minus  = copy.copy(par_dum)
    # par_minus.S_C = par_dum.S_C - gd.d_S_C
    # par_minus = par_calculator(par_minus)
    # sol_minus  = solution_class()
    # sol_minus  = sol_obj_maker(sol_minus, par_minus)
    # sol_minus = nondim_all_solver(sol_minus, par_minus)
    # WT_pure_model, C_pure_model, WT_mix_model, C_mix_model = model_matrix_maker(sol_minus, par_minus)
    # cost_minus = total_cost_calc(WT_pure_exp, WT_pure_model, C_pure_exp, C_pure_model, WT_mix_exp, WT_mix_model, C_mix_exp, C_mix_model)
    
    # grad['S_C'] = (cost_plus - cost_minus) / (2 * gd.d_S_C)
    # # S_C
    
    
    # V_S
    par_plus  = params_class()
    par_plus  = copy.copy(par_dum)
    par_plus.V_S = par_dum.V_S + gd.d_V_S
    par_plus = par_calculator(par_plus)
    sol_plus  = solution_class()
    sol_plus  = sol_obj_maker(sol_plus, par_plus)
    sol_plus = nondim_all_solver(sol_plus, par_plus)
    WT_pure_model, C_pure_model, WT_mix_model, C_mix_model = model_matrix_maker(sol_plus, par_plus)
    cost_plus = total_cost_calc(WT_pure_exp, WT_pure_model, C_pure_exp, C_pure_model, WT_mix_exp, WT_mix_model, C_mix_exp, C_mix_model, sol_plus, par_plus)
    
    par_minus  = params_class()
    par_minus  = copy.copy(par_dum)
    par_minus.V_S = par_dum.V_S - gd.d_V_S
    par_minus = par_calculator(par_minus)
    sol_minus  = solution_class()
    sol_minus  = sol_obj_maker(sol_minus, par_minus)
    sol_minus = nondim_all_solver(sol_minus, par_minus)
    WT_pure_model, C_pure_model, WT_mix_model, C_mix_model = model_matrix_maker(sol_minus, par_minus)
    cost_minus = total_cost_calc(WT_pure_exp, WT_pure_model, C_pure_exp, C_pure_model, WT_mix_exp, WT_mix_model, C_mix_exp, C_mix_model, sol_minus, par_minus)
    
    grad['V_S'] = (cost_plus - cost_minus) / (2 * gd.d_V_S)
    print("Grad V_S : "+str(grad['V_S']))
    # V_S
    
    
    
    # # R_cd
    # par_plus  = params_class()
    # par_plus  = copy.copy(par_dum)
    # par_plus.R_cd = par_dum.R_cd + gd.d_R_cd
    # par_plus = par_calculator(par_plus)
    # sol_plus  = solution_class()
    # sol_plus  = sol_obj_maker(sol_plus, par_plus)
    # sol_plus = nondim_all_solver(sol_plus, par_plus)
    # WT_pure_model, C_pure_model, WT_mix_model, C_mix_model = model_matrix_maker(sol_plus, par_plus)
    # cost_plus = total_cost_calc(WT_pure_exp, WT_pure_model, C_pure_exp, C_pure_model, WT_mix_exp, WT_mix_model, C_mix_exp, C_mix_model, sol_plus, par_plus)
    
    # par_minus  = params_class()
    # par_minus  = copy.copy(par_dum)
    # par_minus.R_cd = par_dum.R_cd - gd.d_R_cd
    # par_minus = par_calculator(par_minus)
    # sol_minus  = solution_class()
    # sol_minus  = sol_obj_maker(sol_minus, par_minus)
    # sol_minus = nondim_all_solver(sol_minus, par_minus)
    # WT_pure_model, C_pure_model, WT_mix_model, C_mix_model = model_matrix_maker(sol_minus, par_minus)
    # cost_minus = total_cost_calc(WT_pure_exp, WT_pure_model, C_pure_exp, C_pure_model, WT_mix_exp, WT_mix_model, C_mix_exp, C_mix_model, sol_minus, par_minus)
    
    # grad['R_cd'] = (cost_plus - cost_minus) / (2 * gd.d_R_cd)
    # print("Grad R_cd : "+str(grad['R_cd']))
    # # R_cd
    
    # # R_a
    # par_plus  = params_class()
    # par_plus  = copy.copy(par_dum)
    # par_plus.R_a = par_dum.R_a + gd.d_R_a
    # par_plus = par_calculator(par_plus)
    # sol_plus  = solution_class()
    # sol_plus  = sol_obj_maker(sol_plus, par_plus)
    # sol_plus = nondim_all_solver(sol_plus, par_plus)
    # WT_pure_model, C_pure_model, WT_mix_model, C_mix_model = model_matrix_maker(sol_plus, par_plus)
    # cost_plus = total_cost_calc(WT_pure_exp, WT_pure_model, C_pure_exp, C_pure_model, WT_mix_exp, WT_mix_model, C_mix_exp, C_mix_model, sol_plus, par_plus)
    
    # par_minus  = params_class()
    # par_minus  = copy.copy(par_dum)
    # par_minus.R_a = par_dum.R_a - gd.d_R_a
    # par_minus = par_calculator(par_minus)
    # sol_minus  = solution_class()
    # sol_minus  = sol_obj_maker(sol_minus, par_minus)
    # sol_minus = nondim_all_solver(sol_minus, par_minus)
    # WT_pure_model, C_pure_model, WT_mix_model, C_mix_model = model_matrix_maker(sol_minus, par_minus)
    # cost_minus = total_cost_calc(WT_pure_exp, WT_pure_model, C_pure_exp, C_pure_model, WT_mix_exp, WT_mix_model, C_mix_exp, C_mix_model, sol_minus, par_minus)
    
    # grad['R_a'] = (cost_plus - cost_minus) / (2 * gd.d_R_a)
    # print("Grad R_a : "+str(grad['R_a']))
    # # R_a
    
    # # R_cd_log
    # par_plus  = params_class()
    # par_plus  = copy.copy(par_dum)
    # par_plus.R_cd_log = par_dum.R_cd_log + gd.d_R_cd_log
    # par_plus = par_calculator(par_plus)
    # sol_plus  = solution_class()
    # sol_plus  = sol_obj_maker(sol_plus, par_plus)
    # sol_plus = nondim_all_solver(sol_plus, par_plus)
    # WT_pure_model, C_pure_model, WT_mix_model, C_mix_model = model_matrix_maker(sol_plus, par_plus)
    # cost_plus = total_cost_calc(WT_pure_exp, WT_pure_model, C_pure_exp, C_pure_model, WT_mix_exp, WT_mix_model, C_mix_exp, C_mix_model, sol_plus, par_plus)
    
    # par_minus  = params_class()
    # par_minus  = copy.copy(par_dum)
    # par_minus.R_cd_log = par_dum.R_cd_log - gd.d_R_cd_log
    # par_minus = par_calculator(par_minus)
    # sol_minus  = solution_class()
    # sol_minus  = sol_obj_maker(sol_minus, par_minus)
    # sol_minus = nondim_all_solver(sol_minus, par_minus)
    # WT_pure_model, C_pure_model, WT_mix_model, C_mix_model = model_matrix_maker(sol_minus, par_minus)
    # cost_minus = total_cost_calc(WT_pure_exp, WT_pure_model, C_pure_exp, C_pure_model, WT_mix_exp, WT_mix_model, C_mix_exp, C_mix_model, sol_minus, par_minus)
    
    # grad['R_cd_log'] = (cost_plus - cost_minus) / (2 * gd.d_R_cd_log)
    # print("Grad R_cd_log : "+str(grad['R_cd_log']))
    # # R_cd_log
    
    # # R_a_log
    # par_plus  = params_class()
    # par_plus  = copy.copy(par_dum)
    # par_plus.R_a_log = par_dum.R_a_log + gd.d_R_a_log
    # par_plus = par_calculator(par_plus)
    # sol_plus  = solution_class()
    # sol_plus  = sol_obj_maker(sol_plus, par_plus)
    # sol_plus = nondim_all_solver(sol_plus, par_plus)
    # WT_pure_model, C_pure_model, WT_mix_model, C_mix_model = model_matrix_maker(sol_plus, par_plus)
    # cost_plus = total_cost_calc(WT_pure_exp, WT_pure_model, C_pure_exp, C_pure_model, WT_mix_exp, WT_mix_model, C_mix_exp, C_mix_model, sol_plus, par_plus)
    
    # par_minus  = params_class()
    # par_minus  = copy.copy(par_dum)
    # par_minus.R_a_log = par_dum.R_a_log - gd.d_R_a_log
    # par_minus = par_calculator(par_minus)
    # sol_minus  = solution_class()
    # sol_minus  = sol_obj_maker(sol_minus, par_minus)
    # sol_minus = nondim_all_solver(sol_minus, par_minus)
    # WT_pure_model, C_pure_model, WT_mix_model, C_mix_model = model_matrix_maker(sol_minus, par_minus)
    # cost_minus = total_cost_calc(WT_pure_exp, WT_pure_model, C_pure_exp, C_pure_model, WT_mix_exp, WT_mix_model, C_mix_exp, C_mix_model, sol_minus, par_minus)
    
    # grad['R_a_log'] = (cost_plus - cost_minus) / (2 * gd.d_R_a_log)
    # print("Grad R_a_log : "+str(grad['R_a_log']))
    # # R_a_log
    
    
    return grad

def sol_obj_maker(sol_dum, par_dum):

    ## mesh
    sol_dum.n_interval_G1 = 50
    sol_dum.n_interval_S  = 50
    sol_dum.dx_G1 = np.pi / sol_dum.n_interval_G1
    sol_dum.dx_S  = np.pi / sol_dum.n_interval_S
    sol_dum.n_mesh_G1 = sol_dum.n_interval_G1 + 1 # both borders, finite difference
    sol_dum.n_mesh_S  = sol_dum.n_interval_S  + 1 # both borders, finite difference
    
    # sol_dum.x_G1 = np.linspace(0 + sol_dum.dx_G1/2 , np.pi-sol_dum.dx_G1/2 , sol_dum.n_mesh_G1)
    # sol_dum.x_S  = np.linspace(np.pi + sol_dum.dx_S/2 , 2*np.pi-sol_dum.dx_S/2 , sol_dum.n_mesh_S)
    sol_dum.x_G1 = np.linspace(0, np.pi, sol_dum.n_mesh_G1)
    sol_dum.x_S  = np.linspace(np.pi, 2*np.pi, sol_dum.n_mesh_S)
    ## mesh
    
    max_Courant = 0.5
    
    # max_v_g1 = par_dum.F_c
    # max_v_s  = par_dum.V_S
    
    max_v_g1 = par_dum.V_0 + par_dum.alpha_w * par_dum.MU_C
    max_v_s  = par_dum.V_S
    
    if max_v_g1<0 or max_v_s<0:
        print("negative velocities!")
        print(vars(par_dum))
        print("###################################")
    
    dt_max = np.min([ max_Courant*sol_dum.dx_G1/max_v_g1, max_Courant*sol_dum.dx_S/max_v_s    ])
    
    
    sol_dum.t_tilde_max = 2.5
    sol_dum.time_tilde = np.linspace(0, sol_dum.t_tilde_max, int(sol_dum.t_tilde_max/dt_max) + 2)
    sol_dum.L = len(sol_dum.time_tilde)
    sol_dum.dt = sol_dum.time_tilde[1]-sol_dum.time_tilde[0]
    
    return sol_dum

def three_unknown_finder(par_dum):
    
    # def equations(vars):
    #     m, R_0_d , V_S = vars
        
        
    #     # Define each equation (left-hand side = 0)
    #     eq1 = (1+R_0_d) * (1-np.exp(-np.pi/V_S) ) / (np.exp(m*np.pi) - 1) + 1 - R_0_d * (1-par_dum.f_D_W) / par_dum.f_D_W
    #     eq2 = m + 1/V_S - np.log(2)/np.pi
    #     eq3 = par_dum.gamma * (par_dum.alpha_c+par_dum.K_C) * (m * par_dum.alpha_w) / ( par_dum.alpha_c * (par_dum.alpha_w+par_dum.K_W) * (1+R_0_d) ) + par_dum.gamma/V_S - np.log(2)/np.pi
        
    #     return [eq1, eq2, eq3]
    
    def equations_2(vars):
        r, eta , zeta = vars
        
        
        # Define each equation (left-hand side = 0)
        eq1 = (1+r) * (1-zeta) / (eta - 1) + 1 - r * (1-par_dum.f_D_W) / par_dum.f_D_W
        eq2 = eta - 2.0 * zeta
        eq3 = par_dum.gamma * (par_dum.alpha_c+par_dum.K_C) *  par_dum.alpha_w * np.log(eta) / ( par_dum.alpha_c * (1+r) * (par_dum.alpha_w+par_dum.K_W)  ) - par_dum.gamma * np.log(zeta) - np.log(2)
        
        return [eq1, eq2, eq3]
    
    # r_guess = 0.3
    # v_s_guess =  (np.pi / 15) / par_dum.beta_wt
    # m_guess = 0.5 * np.log(2) / np.pi
    # initial_guess = (m_guess , r_guess , v_s_guess)
    # solution = fsolve(equations, initial_guess)
    
    r_guess = 0.9
    zeta_guess = np.exp(-np.pi/10)
    eta_guess = np.exp( 0.1 * np.pi)
    initial_guess = (r_guess , eta_guess , zeta_guess)
    solution = fsolve(equations_2, initial_guess)

    # Use fsolve to solve the system
    
    
    # solution, infodict, ier, mesg = fsolve(equations, initial_guess, full_output=True)
    # print("Solution:", solution)
    # print("infodict:", infodict)
    # print("ier:", ier)
    # print("mesg:", mesg)

    # m     = solution[0]
    # R_0_d = solution[1]
    # V_S   = solution[2]
    
    r     = solution[0]
    eta   = solution[1]
    zeta  = solution[2]
    
    
    R_0_d = r
    m     = np.log(eta)/np.pi
    V_S   = -np.pi / np.log(zeta)
    
    
    F = (1 + R_0_d) * (par_dum.alpha_w + par_dum.K_W) / (par_dum.alpha_w * m)

    # (1 + R_0_d) * (par_dum.alpha_w + par_dum.K_W) / (par_dum.alpha_w * F) + 1/V_S - np.log(2)/np.pi
    # par_dum.gamma *  (par_dum.alpha_c + par_dum.K_C) / (par_dum.alpha_c * F) + par_dum.gamma/V_S - np.log(2)/np.pi
    
    
    return F, R_0_d , V_S

def par_calculator(par_dum):
    
    par_dum.R_cd  = np.exp(par_dum.R_cd_log)
    par_dum.R_a = np.exp(par_dum.R_a_log)
    
    m = np.log(2)/np.pi - 1 / par_dum.V_S # doubling BC of WT
    m_prime = 1 / par_dum.V_S
    q = np.log(2)/np.pi - par_dum.gamma / par_dum.V_S # doubling BC of C
    q_prime = par_dum.gamma / par_dum.V_S
    
    par_dum.m = m
    par_dum.m_prime = m_prime
    par_dum.q = q
    par_dum.q_prime = q_prime
    
    
    X = (1-np.exp(-m_prime*np.pi)) / (np.exp(m*np.pi) - 1)
    Y = (1-par_dum.f_D_W)/par_dum.f_D_W
    par_dum.R_0_d = (1 + X) / (Y - X)
    
    if hasattr(par_dum, 'S_W'):
        Z = (1+par_dum.R_0_d)/(m*par_dum.S_W)
        par_dum.MU_W = Z / (1-Z)
    elif hasattr(par_dum, 'MU_W'):
        par_dum.S_W = (1+par_dum.MU_W)*(1+par_dum.R_0_d)/(m*1+par_dum.MU_W)
    
    if hasattr(par_dum, 'S_C') and hasattr(par_dum, 'V_0'):
        if par_dum.S_C>0:
            H = (par_dum.gamma/q - par_dum.V_0)/par_dum.S_C
            par_dum.MU_C = par_dum.gamma * H / (1-H)
        elif par_dum.S_C == 0:
            par_dum.MU_C = 1*par_dum.MU_W
            par_dum.V_0   = par_dum.gamma / q
        
    elif hasattr(par_dum, 'MU_C') and hasattr(par_dum, 'V_0'):
        par_dum.S_C = (par_dum.gamma/q - par_dum.V_0)*(par_dum.gamma+par_dum.MU_C)/par_dum.MU_C
    
    
    # alpha_w
    par_dum.alpha_w = par_dum.S_W / (1+par_dum.MU_W)
    # alpha_w
    
    # alpha_c
    par_dum.alpha_c = par_dum.S_C / (par_dum.gamma+par_dum.MU_C)
    # alpha_c
    
    par_dum.A = np.exp(-par_dum.m*np.pi)*(par_dum.MU_W*par_dum.alpha_w)/par_dum.V_S
    par_dum.B = np.exp(-par_dum.q*np.pi)*(par_dum.V_0 + par_dum.MU_C*par_dum.alpha_c)/par_dum.V_S
    
    # F, R_0_d , V_S = three_unknown_finder(par_dum)
    
    # par_dum.V_S = V_S
    # par_dum.F_c =  ( (par_dum.alpha_c + par_dum.K_C) /par_dum.alpha_c ) * (np.log(2)/(np.pi*par_dum.gamma) -1 / par_dum.V_S )**(-1) # doubling BC of Cancer
    # par_dum.F_w =  (1+par_dum.R_0_d)*(par_dum.alpha_w+par_dum.K_W)/(par_dum.alpha_w * m )
    
    # par_dum.R_0_d = -1 + (np.log(2)/np.pi -1 / par_dum.V_S ) * par_dum.alpha_w * par_dum.F / (par_dum.alpha_w + par_dum.K_W)
    
    
    # par.V_S = np.pi / ((1-par.g_W) * par.T_d_W * par.beta_wt)  # dimensionless
    # par.R_0_d = par.f_D_W / par.f_G1_W
    # par.F =  ( (par.alpha_w + par.K_W) / par.alpha_w ) * \
    #          (1+par.R_0_d) / \
    #          (np.log(2)/np.pi - 1/par.V_S)
    
    # plt.figure()
    # x_plot = np.linspace(0, 5 * par_dum.alpha_w , 1000)
    # # v_g_w = par_dum.F_w * par_dum.alpha_w / (par_dum.alpha_w + par_dum.K_W)
    # # v_g_c = par_dum.F_c * par_dum.alpha_c / (par_dum.alpha_c + par_dum.K_C)
    # v_g_w = par_dum.MU_W * par_dum.alpha_w
    # v_g_c = par_dum.V_0  + par_dum.MU_C * par_dum.alpha_c
    # plt.axhline(y = v_g_w,       xmin=0, xmax=1, color='m', linestyle='--')
    # plt.axhline(y = v_g_c,       xmin=0, xmax=1, color='g', linestyle='--')
    # plt.axhline(y = par_dum.V_S, xmin=0, xmax=1, color='k', linestyle='--')
    # # y_plot_w = par_dum.F_w *x_plot/(x_plot + par_dum.K_W)
    # # y_plot_c = par_dum.F_c *x_plot/(x_plot + par_dum.K_C)
    # y_plot_w = par_dum.MU_W * x_plot
    # y_plot_c = par_dum.V_0  + par_dum.MU_C * x_plot
    # plt.plot(x_plot, y_plot_w, color='m')
    # plt.plot(x_plot, y_plot_c, color='g')
    # # plt.plot(x_plot, y_plot_c-y_plot_w, color='b')
    # plt.axvline(x=par_dum.alpha_w, color='m', ymin = 0, ymax = 1, linestyle='--')
    # plt.axvline(x=par_dum.alpha_c, color='g', ymin = 0, ymax = 1, linestyle='--')
    # plt.tight_layout()
    
    v_s     = par_dum.V_S * par_dum.beta_wt
    v_g1_eq_w = par_dum.MU_W * par_dum.alpha_w * par_dum.beta_wt
    T_w_pure = np.pi / v_s + np.pi / v_g1_eq_w
    g_calculated = (np.pi / v_g1_eq_w) / T_w_pure
    
    par_dum.g_calc = g_calculated
    
    
    # sjhkgshg
    return par_dum

def par_obj_maker(par_dum):
    
    par_dum.beta_wt = 0.0284
    par_dum.beta_c  = 0.0398
    par_dum.gamma = par_dum.beta_c / par_dum.beta_wt
    
    par_dum.f_D_W = 0.2
    par_dum.f_P_W = 1 - par_dum.f_D_W
    # par.f_G1_W = 0.6
    # par.f_S_W = par.f_P_W - par.f_G1_W
    
    par_dum.f_D_W_std = 0.05
    
    
    # par.T_d_W = 20 #doubling time
    # par.T_d_C = 17.4 #doubling time
    # par.g_W = 0.4 # fraction of cell div time in G1
    
    # par.R_0_d = 0.51
    
    # par.V_S = np.pi / ((1-par.g_W) * par.T_d_W * par.beta_wt) # dimensionless
    
    par_dum.t_eval = 60
    
    # g = 0.45
    # par_dum.V_S   = (np.pi/ ((1-g)*20))/par_dum.beta_wt
    
    
    # par_dum.V_S   = 10.79
    par_dum.V_S   = 14.7
    
    # keep one of these
    # par_dum.S_W  = 19.57
    par_dum.S_W  = 16.0
    
    # keep one of these
    
    # keep two of these
    # par_dum.S_C  = 19.57
    par_dum.S_C  = 16.0
    
    
    par_dum.V_0   = 0
    # keep two of these
    
        
    # par_dum.R_a_log  = np.log(4.662931122735371e-06)
    par_dum.R_a_log  = np.log(4.662931122735371e-100)
    # par_dum.R_cd_log = np.log(5.001004700038298e-06)
    par_dum.R_cd_log = np.log(5.001004700038298e-100)
    # par_dum.R_cd_log = np.log(5.001004700038298e-03)
    
    # par.muF_0 = 0.07659941923553702
    
    # par.frac = 0.8
    
    par_dum.n_org = 500
    
    return par_dum

def gd_obj_maker(gd, par_dum):
    
    gd.cost_hist = []
    gd.vals_hist = []
    
    gd.lr = 1
    gd.n_iter = 5000000
    # gd.dh = 0.04
    # scale_muF_0 = 20
    # scale_R = 1e4
    
    # gd.d_K_P = 0.02*par.K_P
    # gd.d_K_C = 0.02*par.K_C
    # gd.d_S   = 0.02*par.S
    # gd.d_muF_0 = 0.02*par.muF_0
    
    # # gd.d_R_0_d = 0.02*par.R_0_d
    # gd.d_R_pd  = 0.02*par.R_pd
    # gd.d_R_a = 0.02*par.R_a
    # gd.d_R_b = 0.02*par.R_b
    

    # gd.list_of_K_P = [par.K_P]
    # gd.list_of_K_C = [par.K_P]
    # gd.list_of_S = [par.S]
    # gd.list_of_muF_0 = [par.muF_0]
    
    # gd.list_of_R_0_d = [par.R_0_d]
    # gd.list_of_R_pd  = [par.R_pd]
    # gd.list_of_R_a = [par.R_a]
    # gd.list_of_R_b = [par.R_b]
    
    # gd.d_K_W = 0.02*par_dum.K_W
    # gd.d_K_C = 0.02*par_dum.K_C
    # gd.d_S   = 0.02*par_dum.S
    
    gd.d_S_W = 0.001*par_dum.S_W
    gd.d_S_C = 0.001*par_dum.S_C
    
    gd.d_S  = 0.001*par_dum.S_W
    
    gd.d_V_S   = 0.001*par_dum.V_S
    
    # gd.d_R_cd  = 0.02*par_dum.R_cd
    # gd.d_R_a =   0.02*par_dum.R_a
    gd.d_R_cd_log  = 0.1*np.abs(par_dum.R_cd_log)
    gd.d_R_a_log =   0.1*np.abs(par_dum.R_a_log)
    # gd.d_R_b = 0.02*par.R_b
    

    # gd.list_of_K_W = [par_dum.K_W]
    # gd.list_of_K_C = [par_dum.K_C]
    # gd.list_of_S = [par_dum.S]
    # # gd.list_of_muF_0 = [par.muF_0]
    
    # # gd.list_of_R_0_d = [par.R_0_d]
    # gd.list_of_R_cd  = [par_dum.R_cd]
    # gd.list_of_R_a = [par_dum.R_a]
    # # gd.list_of_R_b = [par.R_b]
    
    return gd

#begin

WT_pure_exp = np.loadtxt("exp_data"+"/"+"overal_WT_pure.csv", delimiter=',')
C_pure_exp = np.loadtxt("exp_data"+"/"+"overal_C_pure.csv", delimiter=',')
WT_mix_exp = np.loadtxt("exp_data"+"/"+"WT_bar_mix_overal.csv", delimiter=',')
C_mix_exp = np.loadtxt("exp_data"+"/"+"C_bar_mix_overal.csv", delimiter=',')

sol = solution_class()
par = params_class()
gd  = gd_hist_class()

par = par_obj_maker(par)
par = par_calculator(par)
sol = sol_obj_maker(sol, par)
gd  = gd_obj_maker(gd, par)


mixed_sample_bank = np.loadtxt("mixed_sample_bank.csv", delimiter=',', dtype=int)
size = np.shape(mixed_sample_bank)[0]
sample_indices_mix = np.random.randint(0,size,par.n_org)
np.savetxt('sample_indices_mix.csv', X=sample_indices_mix, delimiter=',', fmt='%d')


# dsvd



###################### INITIAL ############################
# par.K_W = 2.503114460981055 #nondim (= F_0*k_w)
# par.K_C = 0.7253978328301385 #nondim (= F_0*k_c)
# par.S   = 0.9352886155074336 #nondim (= s / mu)
# par.muF_0 = 0.07659941923553702
# par.R = 2.7e-4

# K_W = 2.689240915637694 #nondim (= F_0*k_w)
# K_C = 0.8180592584163358 #nondim (= F_0*k_c)
# S   = 0.7301552976980761 #nondim (= s / mu)
# muF_0 = 0.09141949831898115
# R = 1.0e-4



sol = nondim_all_solver(sol, par)
WT_pure_model, C_pure_model, WT_mix_model, C_mix_model = model_matrix_maker(sol, par)
cost = total_cost_calc(WT_pure_exp, WT_pure_model, C_pure_exp, C_pure_model, WT_mix_exp, WT_mix_model, C_mix_exp, C_mix_model, sol, par)
gd.cost_hist.append(cost)
# gd.vals_hist.append(copy.copy(par))
# gd.vals_hist.append([par.K_W, par.K_C, par.S, par.V_S].copy())
gd.vals_hist.append([par.S_W, par.S_C, par.V_S, par.R_cd, par.R_a].copy())


sol_population_plotter(par, sol)
compos_effect_plotter(sol, par)
composition_vs_time_plotter(sol, par)

par_vals = []
par_vals.append(par.S_W)
par_vals.append(par.S_C)
par_vals.append(par.V_S)
par_vals.append(par.MU_W)
par_vals.append(par.MU_C)
par_vals.append(par.alpha_w)
par_vals.append(par.alpha_c)
par_vals.append(par.R_0_d)
np.savetxt('par_vals.csv', par_vals, delimiter=',')





