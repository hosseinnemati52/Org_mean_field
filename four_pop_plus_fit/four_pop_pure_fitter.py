# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 12:37:08 2024

@author: Nemat002
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import ast
import matplotlib.animation as animation
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import linregress


def load_excel_sheet(file_name, sheet_name):
    """
    Loads a sheet from an Excel file and reads the data written in columns.

    :param file_path: Path to the Excel file.
    :param sheet_name: Name of the sheet to load.
    :return: A DataFrame containing the data from the specified sheet.
    """
    try:
        # Load the sheet into a DataFrame
        df = pd.read_excel(file_name, sheet_name=sheet_name)
        
        # Print the column names
        print("Columns:", df.columns.tolist())
        
        # Print the data in columns
        for column in df.columns:
            
            if column=='y':
                y_avg = np.array(df[column])
            elif column=='y_err':
                y_err = np.array(df[column])
            elif column=='x':
                t = np.array(df[column])
            
        return t, y_avg, y_err
    except Exception as e:
        print(f"An error occurred: {e}")

def plotter(x, y, X, X_err, Y, Y_err, title, save_switch):
    
    plt.figure()
    plt.plot(x, y, label='model')
    plt.errorbar(X, Y, xerr=X_err, yerr=Y_err, fmt='o', ecolor='r', capsize=2, label='Exp')

    # Add labels and title
    plt.xlabel('h')
    plt.ylabel(r'$N_{cells}/N_0$')
    plt.title(title)
    plt.legend()
    if save_switch:
        plt.savefig(title+'.PNG', dpi=300)
        
    return

def fit_line(x, y, y_max, y_min,  title, save_switch):
    """
    Fits a line to data points and plots the original data along with the fitted line.

    :param x: Array-like, the x coordinates of the data points.
    :param y: Array-like, the y coordinates of the data points.
    :return: Tuple containing the slope and intercept of the fitted line.
    """
    # Fit a line using np.polyfit
    slope, intercept = np.polyfit(x, y, 1)
    
    # Generate the y values of the fitted line
    y_fit = slope * np.array(x) + intercept
    
    plt.figure()
    # Plot the original data points
    # plt.scatter(x, y, label='Data points', color='blue')
    
    # Asymmetric error bars for y
    yerr = [y - y_min, y_max - y]
    
    # Create scatter plot with asymmetric error bars in the y-direction
    plt.errorbar(x, y, yerr=yerr, fmt='o', ecolor='blue', capsize=3,  label='Data points', color='blue')
    
    # Add labels and title
    
    # Plot the fitted line
    plt.plot(x, y_fit, label=f'Fitted line: y = {slope:.4f}x + {intercept:.2f}', color='red')
    
    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend()
    
    if save_switch:
        plt.savefig(title+'.PNG', dpi=300)
    
    return slope, intercept

def avg_err_extractor(data):
    
    t_exp = data[:,0]
    Y_exp = 0.0 * t_exp
    Y_exp_err = 0.0 * t_exp
    Y_exp_std = 0.0 * t_exp
    
    for i in range(len(t_exp)):
        row_data = data[i,1:]
        ratio_data = row_data / data[0,1:]
        
        mask = ratio_data[~np.isnan(ratio_data)]
        
        # print(len(mask))
        
        Y_exp[i] = np.mean(mask)
        Y_exp_err[i] = np.std(mask)/np.sqrt(len(mask))
        Y_exp_std[i] = np.std(mask)
        
    return Y_exp, Y_exp_err, Y_exp_std

def exponential_fitting_b0(data, sheet_name):
    
    t_exp = data[:,0]
    Y_exp = 0.0 * t_exp
    Y_exp_err = 0.0 * t_exp
    
    for i in range(len(t_exp)):
        row_data = data[i,1:]
        ratio_data = row_data / data[0,1:]
        
        mask = ratio_data[~np.isnan(ratio_data)]
        
        Y_exp[i] = np.mean(mask)
        Y_exp_err[i] = np.std(mask)/np.sqrt(len(mask))
    
    # slope, intercept = fit_line(t_exp, np.log(Y_exp), np.log(Y_exp+Y_exp_err), np.log(Y_exp-Y_exp_err), sheet_name+'_ylog', 1)
    x_fit = t_exp.copy()
    y_fit = np.log(Y_exp).copy()
    # yerr_fit = 0 * Y_exp_err
    yerr_fit = Y_exp_err.copy()
    # threshold = np.mean(Y_exp_err) * (1e-7)
    threshold =  (1e-7)
    for i in range(len(yerr_fit)):
        if i == 0:
            yerr_fit[i]  = 0
        elif Y_exp_err[i] < threshold:
            # yerr_fit[i]  = 0.5 * ( (np.log(Y_exp[i]+threshold)-np.log(Y_exp[i])) + (np.log(Y_exp[i]) - np.log(Y_exp[i]-threshold)) )
            yerr_fit[i] = Y_exp_err[i]/Y_exp[i]
        else:
            # yerr_fit[i]  = 0.5 * ( (np.log(Y_exp[i]+Y_exp_err[i])-np.log(Y_exp[i])) + (np.log(Y_exp[i]) - np.log(Y_exp[i]-Y_exp_err[i])) )
            # yerr_fit[i]  = np.max ([ (np.log(Y_exp[i]+Y_exp_err[i])-np.log(Y_exp[i])) + (np.log(Y_exp[i]) - np.log(Y_exp[i]-Y_exp_err[i])) ])
            yerr_fit[i] = Y_exp_err[i]/Y_exp[i]
    
    # slope, intercept = fit_line(x_fit, y_fit, y_fit+yerr_fit, y_fit-yerr_fit, sheet_name+'_ylog', 1)
    # fitter_0_0()
    yerr_fit[0]  = threshold
    
    np.savetxt('overal_'+sheet_name+'.csv', X=np.array([t_exp, Y_exp, Y_exp_err]), delimiter=',', fmt='%.8f')
    
    #########################################
    ## fitting with no uncertainty
    # weights = 1/yerr_fit**2
    # slope_numerator = 0.
    # slope_denominator = 0.
    # for i in range(len(yerr_fit)):
    #     slope_numerator   += weights[i] * x_fit[i] * y_fit[i]
    #     slope_denominator += weights[i] * x_fit[i] * x_fit[i]
    # slope = slope_numerator/slope_denominator
    #########################################
    
    
    
    ######################################
    # fitting with uncertainty
    x_lin_fit =  x_fit
    y_lin_fit =  y_fit
    y_lin_ERR_fit =  yerr_fit
    
    Delta_mat =     np.array([[0., 0.], [0., 0.]])
    intercept_mat = np.array([[0., 0.], [0., 0.]])
    slope_mat =     np.array([[0., 0.], [0., 0.]])
    
    Delta_mat[0,0] = np.sum(1/y_lin_ERR_fit**2)
    Delta_mat[0,1] = np.sum(x_lin_fit/y_lin_ERR_fit**2)
    Delta_mat[1,0] = np.sum(x_lin_fit/y_lin_ERR_fit**2)
    Delta_mat[1,1] = np.sum(x_lin_fit**2/y_lin_ERR_fit**2)
    Delta = np.linalg.det(Delta_mat)
    
    intercept_mat[0,0] = np.sum(y_lin_fit/y_lin_ERR_fit**2)
    intercept_mat[0,1] = np.sum(x_lin_fit/y_lin_ERR_fit**2)
    intercept_mat[1,0] = np.sum(x_lin_fit*y_lin_fit/y_lin_ERR_fit**2)
    intercept_mat[1,1] = np.sum(x_lin_fit**2/y_lin_ERR_fit**2)
    intercept = (1/Delta)*np.linalg.det(intercept_mat)
    
    slope_mat[0,0] = np.sum(1/y_lin_ERR_fit**2)
    slope_mat[0,1] = np.sum(y_lin_fit/y_lin_ERR_fit**2)
    slope_mat[1,0] = np.sum(x_lin_fit/y_lin_ERR_fit**2)
    slope_mat[1,1] = np.sum(x_lin_fit*y_lin_fit/y_lin_ERR_fit**2)
    slope = (1/Delta)*np.linalg.det(slope_mat)
    
    
    intercept_ERR = np.sqrt(  (1/Delta) * np.sum(x_lin_fit**2/y_lin_ERR_fit**2) )
    slope_ERR =     np.sqrt(  (1/Delta) * np.sum(1.0         /y_lin_ERR_fit**2) )
    ######################################
    
    
    
    
    alpha = slope #Wild type cells (healthy cells)
    Aw_0 = 1.0
    # t = np.linspace(0, 70, 71)
    t = np.linspace(t_exp[0], t_exp[-1], 71)
    Aw = Aw_0 * np.exp(alpha * t)
    plotter(t, Aw/Aw_0, t_exp, 0*t_exp, Y_exp, Y_exp_err, sheet_name, 1)
    
    plt.figure()
    plt.plot(t, slope * t , label=f'Fitted line: y = {slope:.4f}x\nSlope uncertainty: ±{slope_ERR:.4f}')
    plt.errorbar(t_exp, np.log(Y_exp), xerr=0*t_exp, yerr=yerr_fit, fmt='o', ecolor='r', capsize=2, label='Exp')
    plt.xlabel('h')
    plt.ylabel(r'$\log{N_{cells}/N_0}$')
    plt.legend()
    plt.title(sheet_name)
    plt.tight_layout()
    plt.savefig(sheet_name+'_log.PNG', dpi=300)
    
    # plt.figure()
    # plt.plot(t, 0 * t , label='model')
    # plt.errorbar(t_exp, np.log(Y_exp)-slope*t_exp, xerr=0*t_exp, yerr=yerr_fit, fmt='o', ecolor='r', capsize=2, label='Exp')
    # plt.xlabel('h')
    # plt.ylabel(r'$\log{N_{cells}/N_0}$-fit')
    # plt.legend()
    # plt.title(sheet_name)
    # plt.tight_layout()
    # plt.savefig(sheet_name+'_log_diff.PNG', dpi=300)
    
    exponent = slope
    exponent_ERR = slope_ERR
    
    return exponent, exponent_ERR

def WLS_fitter(x_fit, y_fit, yerr_fit):
    
    ######################################
    # fitting with uncertainty
    x_lin_fit =  x_fit
    y_lin_fit =  y_fit
    y_lin_ERR_fit =  yerr_fit
    
    Delta_mat =     np.array([[0., 0.], [0., 0.]])
    intercept_mat = np.array([[0., 0.], [0., 0.]])
    slope_mat =     np.array([[0., 0.], [0., 0.]])
    
    Delta_mat[0,0] = np.sum(1/y_lin_ERR_fit**2)
    Delta_mat[0,1] = np.sum(x_lin_fit/y_lin_ERR_fit**2)
    Delta_mat[1,0] = np.sum(x_lin_fit/y_lin_ERR_fit**2)
    Delta_mat[1,1] = np.sum(x_lin_fit**2/y_lin_ERR_fit**2)
    Delta = np.linalg.det(Delta_mat)
    
    intercept_mat[0,0] = np.sum(y_lin_fit/y_lin_ERR_fit**2)
    intercept_mat[0,1] = np.sum(x_lin_fit/y_lin_ERR_fit**2)
    intercept_mat[1,0] = np.sum(x_lin_fit*y_lin_fit/y_lin_ERR_fit**2)
    intercept_mat[1,1] = np.sum(x_lin_fit**2/y_lin_ERR_fit**2)
    intercept = (1/Delta)*np.linalg.det(intercept_mat)
    
    slope_mat[0,0] = np.sum(1/y_lin_ERR_fit**2)
    slope_mat[0,1] = np.sum(y_lin_fit/y_lin_ERR_fit**2)
    slope_mat[1,0] = np.sum(x_lin_fit/y_lin_ERR_fit**2)
    slope_mat[1,1] = np.sum(x_lin_fit*y_lin_fit/y_lin_ERR_fit**2)
    slope = (1/Delta)*np.linalg.det(slope_mat)
    
    
    intercept_ERR = np.sqrt(  (1/Delta) * np.sum(x_lin_fit**2/y_lin_ERR_fit**2) )
    slope_ERR =     np.sqrt(  (1/Delta) * np.sum(1.0         /y_lin_ERR_fit**2) )
    ######################################
    
    return slope, slope_ERR, intercept, intercept_ERR

def organoids_indiv_plot(WT_data, C_data):
    
    time_list = [] # each element of this list is for an organoid
    WT_list = [] # each element of this list is for an organoid
    C_list = [] # each element of this list is for an organoid
    
    N_org      = len(WT_data[0,1:])
    N_timestep = len(WT_data[:,0])
    
    overal_time = WT_data[:,0]
    WT_bar_dict = dict() # WT_bar(t) = WT(t)/WT(0)
    C_bar_dict = dict() # C_bar(t) = C(t)/C(0)
    tot_bar_dict = dict()  # tot_bar(t) = tot(t)/tot(0)
    
    for col_c in range(1, N_org+1):
        
        time_array = []
        WT_array = []
        C_array = []
        
        WT_bar_array = []
        C_bar_array = []
        tot_bar_array = []
        
        for row_c in range(N_timestep):
            if (~np.isnan(WT_data[row_c, col_c]) and ~np.isnan(C_data[row_c, col_c])):
                time_array.append(WT_data[row_c, 0])
                WT_array.append(WT_data[row_c, col_c])
                C_array.append(C_data[row_c, col_c])
                
                try:
                    WT_bar_dict[row_c].append(WT_data[row_c, col_c]/ WT_data[0, col_c])
                except:
                    WT_bar_dict[row_c] = [WT_data[row_c, col_c]/ WT_data[0, col_c]]
                
                try:
                    C_bar_dict[row_c].append(C_data[row_c, col_c]/ C_data[0, col_c])
                except:
                    C_bar_dict[row_c] = [C_data[row_c, col_c]/ C_data[0, col_c]]
                    
                try:
                    tot_bar_dict[row_c].append( (C_data[row_c, col_c] + WT_data[row_c, col_c]) / (C_data[0, col_c] + WT_data[0, col_c]) )
                except:
                    tot_bar_dict[row_c] = [( (C_data[row_c, col_c] + WT_data[row_c, col_c]) / (C_data[0, col_c] + WT_data[0, col_c]) )]
                
        
        time_array = np.array(time_array)
        WT_array = np.array(WT_array)
        C_array = np.array(C_array)
        
        time_list.append(time_array)
        WT_list.append(WT_array)
        C_list.append(C_array)
        
        
        
    overal_WT_bar = 0.0 * overal_time
    overal_C_bar = 0.0 * overal_time
    overal_tot_bar = 0.0 * overal_time
    
    overal_WT_bar_ERR = 0.0 * overal_time
    overal_C_bar_ERR = 0.0 * overal_time
    overal_tot_bar_ERR = 0.0 * overal_time
    
    for time_c in range(len(overal_time)):
        
        overal_WT_bar[time_c] = np.mean(WT_bar_dict[time_c])
        overal_WT_bar_ERR[time_c] = np.std(WT_bar_dict[time_c])/np.sqrt(len(WT_bar_dict[time_c]))
        
        overal_C_bar[time_c] = np.mean(C_bar_dict[time_c])
        overal_C_bar_ERR[time_c] = np.std(C_bar_dict[time_c])/np.sqrt(len(C_bar_dict[time_c]))
        
        overal_tot_bar[time_c] = np.mean(tot_bar_dict[time_c])
        overal_tot_bar_ERR[time_c] = np.std(tot_bar_dict[time_c])/np.sqrt(len(tot_bar_dict[time_c]))
        
    # N_org = len(time_list)
    
    # plt.figure()
    # # for org_c in range(N_org):
    # for org_c in [15]:
    #     x_plt = time_list[org_c]
        
    #     # y_plt = WT_list[org_c]
    #     y_plt = WT_list[org_c]/WT_list[org_c][0]
    #     plt.scatter(x_plt, y_plt, marker='o', color='m', s=20, label='WT', linestyle='dashed')
    #     plt.plot(x_plt, y_plt, color='m', linestyle='dashed')
        
    #     # y_plt = C_list[org_c]
    #     y_plt = C_list[org_c]/C_list[org_c][0]
    #     plt.scatter(x_plt, y_plt, marker='o', color='g', s=20, label='C', linestyle='dashed')
    #     plt.plot(x_plt, y_plt, color='g', linestyle='dashed')
        
    #     # y_plt = C_list[org_c] + WT_list[org_c]
    #     y_plt = (C_list[org_c] + WT_list[org_c]) / (C_list[org_c][0] + WT_list[org_c][0]) 
    #     plt.scatter(x_plt, y_plt, marker='o', color='k', s=20, label='tot', linestyle='dashed')
    #     plt.plot(x_plt, y_plt, color='k', linestyle='dashed')
    
    # plt.xlabel('h')
    # # plt.ylabel(r'$N_{cells}/N_0$')
    # plt.title('organoids')
    # plt.legend()
    # plt.savefig('org_indiv'+'.PNG', dpi=300)
    
    
    
    ### bar plot
    plt.figure()
    # plt.scatter(x_plt, y_plt, marker='o', color='m', s=20, label='WT', linestyle='dashed')
    plt.errorbar(overal_time, overal_WT_bar, yerr=overal_WT_bar_ERR, fmt='o', color='m', capsize=2, label='WT')
    plt.plot(overal_time, overal_WT_bar, color='m', linestyle='dashed')
    np.savetxt('overal_WT_bar.csv', X=np.array([overal_time, overal_WT_bar, overal_WT_bar_ERR]), delimiter=',', fmt='%.8f')
    
    plt.errorbar(overal_time, overal_C_bar, yerr=overal_C_bar_ERR, fmt='o', color='g', capsize=2, label='C')
    plt.plot(overal_time, overal_C_bar, color='g', linestyle='dashed')
    np.savetxt('overal_C_bar.csv', X=np.array([overal_time, overal_C_bar, overal_C_bar_ERR]), delimiter=',', fmt='%.8f')
    
    plt.errorbar(overal_time, overal_tot_bar, yerr=overal_tot_bar_ERR, fmt='o', color='k', capsize=2, label='tot')
    plt.plot(overal_time, overal_tot_bar, color='k', linestyle='dashed')
    np.savetxt('overal_tot_bar.csv', X=np.array([overal_time, overal_tot_bar, overal_tot_bar_ERR]), delimiter=',', fmt='%.8f')
    
    plt.xlabel('h')
    plt.ylabel(r'$N_{cells}/N_0$')
    plt.title('bar quantities')
    plt.legend()
    # plt.yscale("log")
    plt.grid()
    plt.savefig('org_bar'+'.PNG', dpi=300)
    ### bar plot
        
    return

def model_3_solver(params, alpha, beta, init_cond, t_exp, C_exp, WT_exp, switch_plot):
    
    gamma, delta = params
    Aw_0 , Ac_0 = init_cond
    
    sq_err_C  = 0.0
    sq_err_WT = 0.0
    
    N_dt = 71
    t_model = np.linspace(t_exp[0], t_exp[-1], N_dt)
    dt = t_model[1] - t_model[0]
    
    Aw   = 0.0 * t_model
    Ac   = 0.0 * t_model
    Atot = 0.0 * t_model
    Aw[0] = Aw_0
    Ac[0] = Ac_0
    Atot[0] = Aw[0] + Ac[0]
    
    t_exp_c = 1
    
    for time_c in range(1,len(t_model)):
        
        diff_Ac = (beta + gamma * Aw[time_c-1] / (Aw[time_c-1] + Ac[time_c-1])) * Ac[time_c-1]
        diff_Aw = alpha * (1.0 - delta * Ac[time_c-1] / (Aw[time_c-1] + Ac[time_c-1])) * Aw[time_c-1]
        
        Ac[time_c] = Ac[time_c-1] + dt * diff_Ac
        Aw[time_c] = Aw[time_c-1] + dt * diff_Aw
        
        Atot[time_c] = Aw[time_c] + Ac[time_c]
        
        if np.abs(t_model[time_c]-t_exp[t_exp_c]) < 1.0e-7 :
            
            sq_err_C  += ((Ac[time_c] -  C_exp[t_exp_c])/C_exp[t_exp_c] ) **2
            sq_err_WT += ((Aw[time_c] - WT_exp[t_exp_c])/WT_exp[t_exp_c] )**2
            
            t_exp_c += 1
    
    
    if switch_plot:
        plt.figure()
        plt.plot(t_model, Ac , label='Ac (model)')
        plt.errorbar(t_exp, C_exp, xerr=0*t_exp, yerr= 0.0 * C_exp, fmt='o', ecolor='r', capsize=2, label='Ac (exp)')
        plt.xlabel('h')
        plt.ylabel(r'$N_{cells}/N_0$')
        plt.legend()
        plt.title('Ac(t) ; '+ r'$\gamma = $'+str(round(gamma, 5))+' ; '+ r'$\delta = $'+str(round(delta, 5)))
        plt.tight_layout()
        plt.savefig('Ac_model_exp.PNG', dpi=300)
        
        plt.figure()
        plt.plot(t_model, Aw , label='Aw (model)')
        plt.errorbar(t_exp, WT_exp, xerr=0*t_exp, yerr= 0.0 * WT_exp, fmt='o', ecolor='r', capsize=2, label='Aw (exp)')
        plt.xlabel('h')
        plt.ylabel(r'$N_{cells}/N_0$')
        plt.legend()
        plt.title('Aw(t) ; '+ r'$\gamma = $'+str(round(gamma, 5))+' ; '+ r'$\delta = $'+str(round(delta, 5)))
        plt.tight_layout()
        plt.savefig('Aw_model_exp.PNG', dpi=300)
        
    return sq_err_C + sq_err_WT

def evaluate_linear_fit(x, y, y_pred):
    """
    Evaluate the quality of a linear fit.

    Parameters:
    - x: array-like, independent variable values
    - y: array-like, actual dependent variable values
    - y_pred: array-like, predicted values from the linear model

    Returns:
    - Dictionary containing R^2, Adjusted R^2, MAE, RMSE, and Residual Normality test.
    """
    # Compute R-squared (Coefficient of Determination)
    ss_total = np.sum((y - np.mean(y))**2)
    ss_residual = np.sum((y - y_pred)**2)
    r2 = 1 - (ss_residual / ss_total)

    return r2

def model_3_err_optimizer(params, alpha, beta, WT_mix_data, C_mix_data, switch_plot):
    
    gamma, delta = params
    
    N_org = len(WT_mix_data[0,1:])
    N_timestep = len(WT_mix_data[:,0])
    
    sq_err_C  = 0.0
    sq_err_WT = 0.0
    
    t_exp = WT_mix_data[:, 0]
    
    for org_c in range(1, N_org+1):
        
        C_exp =   C_mix_data[:, org_c]
        WT_exp = WT_mix_data[:, org_c]
        
        Aw_0 = WT_mix_data[0, org_c]
        Ac_0 =  C_mix_data[0, org_c]
        
        N_dt = 71
        t_model = np.linspace(t_exp[0], t_exp[-1], N_dt)
        dt = t_model[1] - t_model[0]
        
        Aw   = 0.0 * t_model
        Ac   = 0.0 * t_model
        Atot = 0.0 * t_model
        Aw[0] = Aw_0
        Ac[0] = Ac_0
        Atot[0] = Aw[0] + Ac[0]
        
        t_exp_c = 1
    
        for time_c in range(1,len(t_model)):
            
            # diff_Ac = (beta + gamma * Aw[time_c-1] / (Aw[time_c-1] + Ac[time_c-1])) * Ac[time_c-1]
            # diff_Aw = alpha * (1.0 - delta * Ac[time_c-1] / (Aw[time_c-1] + Ac[time_c-1])) * Aw[time_c-1]
            
            diff_Ac = (beta + gamma * Aw[0] / (Aw[0] + Ac[0])) * Ac[time_c-1]
            diff_Aw = alpha * (1.0 - delta * Ac[0] / (Aw[0] + Ac[0])) * Aw[time_c-1]
            
            Ac[time_c] = Ac[time_c-1] + dt * diff_Ac
            Aw[time_c] = Aw[time_c-1] + dt * diff_Aw
            
            Atot[time_c] = Aw[time_c] + Ac[time_c]
            
            if np.abs(t_model[time_c]-t_exp[t_exp_c]) < 1.0e-7 :
                
                if (~np.isnan(C_exp[t_exp_c])):
                    # try:
                    #     sq_err_C  += ((Ac[time_c] -  C_exp[t_exp_c])/C_exp[t_exp_c] ) **2
                    # except:
                        # pass
                    sq_err_C  += ((Ac[time_c] -  C_exp[t_exp_c])/C_exp[t_exp_c] ) **2
                if (~np.isnan(WT_exp[t_exp_c])):
                    # try:
                    #     sq_err_WT += ((Aw[time_c] - WT_exp[t_exp_c])/WT_exp[t_exp_c] )**2
                    # except:
                    #     pass
                    sq_err_WT += ((Aw[time_c] - WT_exp[t_exp_c])/WT_exp[t_exp_c] )**2
                
                t_exp_c += 1
    
    
    if switch_plot:
        plt.figure()
        plt.plot(t_model, Ac , label='Ac (model)')
        plt.errorbar(t_exp, C_exp, xerr=0*t_exp, yerr= 0.0 * C_exp, fmt='o', ecolor='r', capsize=2, label='Ac (exp)')
        plt.xlabel('h')
        plt.ylabel(r'$N_{cells}/N_0$')
        plt.legend()
        plt.title('Ac(t) ; '+ r'$\gamma = $'+str(round(gamma, 5))+' ; '+ r'$\delta = $'+str(round(delta, 5)))
        plt.tight_layout()
        plt.savefig('Ac_model_exp.PNG', dpi=300)
        
        plt.figure()
        plt.plot(t_model, Aw , label='Aw (model)')
        plt.errorbar(t_exp, WT_exp, xerr=0*t_exp, yerr= 0.0 * WT_exp, fmt='o', ecolor='r', capsize=2, label='Aw (exp)')
        plt.xlabel('h')
        plt.ylabel(r'$N_{cells}/N_0$')
        plt.legend()
        plt.title('Aw(t) ; '+ r'$\gamma = $'+str(round(gamma, 5))+' ; '+ r'$\delta = $'+str(round(delta, 5)))
        plt.tight_layout()
        plt.savefig('Aw_model_exp.PNG', dpi=300)
        
    return sq_err_C + sq_err_WT

def alternative_uncertainties(time_data, norm_avg_log, norm_err_log):
    
    norm_avg_log_orig = norm_avg_log.copy()
    
    n_samples = 1000
    slope_samples = np.zeros(n_samples)
    intercept_samples = np.zeros(n_samples)
    
    x = time_data.copy()
    y = norm_avg_log.copy()
    for i in range(n_samples):
        
        for j in range(len(norm_avg_log_orig)):
            y[j] = np.random.uniform(low=norm_avg_log_orig[j]-norm_err_log[j], high=norm_avg_log_orig[j]+norm_err_log[j], size=1)
            # norm_avg_log[j] = np.random.normal(loc=norm_avg_log_orig[j], scale=norm_err_log[j], size=1)
            
        # slope, slope_ERR, intercept, intercept_ERR = WLS_fitter(time_data, norm_avg_log, norm_err_log)
        
        m, b = np.polyfit(x, y, 1)
        
        slope_samples[i] = m
        intercept_samples[i] = b
        
        slope_ERR = np.std(slope_samples)
        intercept_ERR = np.std(intercept_samples)
        
    return slope_ERR , intercept_ERR

## Pure organoids
WT_pure_data = np.genfromtxt('WT_pure.csv', delimiter=',', skip_header=0, filling_values=np.nan)
C_pure_data = np.genfromtxt('C_pure.csv', delimiter=',', skip_header=0, filling_values=np.nan)

time_data = WT_pure_data[:,0]

WT_norm_avg , WT_norm_err , WT_norm_std = avg_err_extractor(WT_pure_data)
C_norm_avg , C_norm_err , C_norm_std    = avg_err_extractor(C_pure_data)
np.savetxt('overal_WT_pure.csv', X=np.array([time_data, WT_norm_avg, WT_norm_err]), delimiter=',', fmt='%.8f')
np.savetxt('overal_C_pure.csv', X=np.array([time_data, C_norm_avg, C_norm_err]), delimiter=',', fmt='%.8f')

WT_norm_avg_log = np.log(WT_norm_avg)
WT_norm_err_log = WT_norm_err / WT_norm_avg
WT_norm_err_log[0] = 1e-7
C_norm_avg_log = np.log(C_norm_avg)
C_norm_err_log = C_norm_err / C_norm_avg
C_norm_err_log[0] = 1e-7

WT_slope, WT_slope_ERR, WT_intercept, WT_intercept_ERR = WLS_fitter(time_data, WT_norm_avg_log, WT_norm_err_log)
C_slope, C_slope_ERR, C_intercept, C_intercept_ERR = WLS_fitter(time_data, C_norm_avg_log, C_norm_err_log)

WT_slope_ERR , WT_intercept_ERR = alternative_uncertainties(time_data, WT_norm_avg_log, WT_norm_std / WT_norm_avg)
C_slope_ERR  , C_intercept_ERR = alternative_uncertainties(time_data, C_norm_avg_log, C_norm_std / C_norm_avg)


pure_organoid_exponents = np.zeros((2,2))
pure_organoid_exponents[0,0] = WT_slope
pure_organoid_exponents[0,1] = C_slope
pure_organoid_exponents[1,0] = WT_slope_ERR
pure_organoid_exponents[1,1] = C_slope_ERR
np.savetxt('pure_organoid_exponents.csv', X= pure_organoid_exponents, fmt='%.8f', delimiter=',')

WT_fit_goodness = evaluate_linear_fit(time_data, WT_norm_avg_log, WT_slope*time_data+WT_intercept)
C_fit_goodness = evaluate_linear_fit(time_data, C_norm_avg_log, C_slope*time_data+C_intercept)
np.savetxt("pure_fit_goodness.csv", delimiter=',',X= [WT_fit_goodness, C_fit_goodness])
## Pure organoids

## mixed organoids
WT_mix_data = np.genfromtxt('WT_mix.csv', delimiter=',', skip_header=0, filling_values=np.nan)
C_mix_data = np.genfromtxt('C_mix.csv', delimiter=',', skip_header=0, filling_values=np.nan)

WT_mix_norm_avg , WT_mix_norm_err , WT_mix_norm_std = avg_err_extractor(WT_mix_data)
C_mix_norm_avg , C_mix_norm_err , C_mix_norm_std    = avg_err_extractor(C_mix_data)
np.savetxt('overal_WT_mix.csv', X=np.array([time_data, WT_mix_norm_avg, WT_mix_norm_err]), delimiter=',', fmt='%.8f')
np.savetxt('overal_C_mix.csv', X=np.array([time_data, C_mix_norm_avg, C_mix_norm_err]), delimiter=',', fmt='%.8f')
## mixed organoids


################# plotting together #####################
plt.figure()
# err1 = plt.errorbar(time_data, WT_norm_avg, xerr=0*time_data, yerr=WT_norm_err, fmt='o', color='m', ecolor='m', capsize=2, label='pure WT')
# err2 =plt.errorbar(time_data, C_norm_avg, xerr=0*time_data, yerr=C_norm_err, fmt='o', color='g', ecolor='g', capsize=2, label='pure C')
err1 = plt.errorbar(time_data, WT_norm_avg,  yerr=WT_norm_err, fmt='o', color='m', ecolor='m', capsize=2, label='pure WT')
err2 =plt.errorbar(time_data, C_norm_avg,  yerr=C_norm_err, fmt='o', color='g', ecolor='g', capsize=2, label='pure C')

err1_handle = err1.lines[0]
err2_handle = err2.lines[0]

time_fit = np.linspace(np.min(time_data),np.max(time_data), 1000)

line3, =plt.plot(time_fit, np.exp(WT_slope*time_fit) , color='m', label='fit: '+r'$\beta = $'+f'{WT_slope:.4f} ±{WT_slope_ERR:.4f}', linestyle='--')
line4, =plt.plot(time_fit, np.exp(C_slope*time_fit) , color='g', label='fit: '+r'$\beta = $'+f'{C_slope:.4f} ±{C_slope_ERR:.4f}', linestyle='--')

# err3 = plt.errorbar(time_data, WT_mix_norm_avg, xerr=0*time_data, yerr=WT_mix_norm_err, fmt='s', color='m', ecolor='m', capsize=2, label='mixed WT', mfc='none')
# err4 =plt.errorbar(time_data, C_mix_norm_avg, xerr=0*time_data, yerr=C_mix_norm_err, fmt='s', color='g', ecolor='g', capsize=2, label='mixed C', mfc='none')
err3 = plt.errorbar(time_data, WT_mix_norm_avg, yerr=WT_mix_norm_err, fmt='s', color='m', ecolor='m', capsize=2, label='mixed WT', mfc='none')
err4 =plt.errorbar(time_data, C_mix_norm_avg, yerr=C_mix_norm_err, fmt='s', color='g', ecolor='g', capsize=2, label='mixed C', mfc='none')

err3_handle = err3.lines[0]
err4_handle = err4.lines[0]

plt.xlabel(r'$t$'+' (h)', fontsize=15)
plt.ylabel(r'$\langle{{N(t)\;/\;N(0)}\rangle}$', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

handles, labels = plt.gca().get_legend_handles_labels()   
# specify order 
order = [2, 3, 4 , 5]
# pass handle & labels lists along with order as below 
plt.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=12) 

# plt.legend(handles=[err1_handle, err2_handle, line3, line4])
# plt.title()
plt.yscale("log")
plt.tight_layout()
plt.savefig('four_pops_pure_fit.PNG', dpi=300)
################# plotting together #####################
