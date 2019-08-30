# coding=utf-8
'''
A Dynamic Model Averaging for the Discovery of Time-Varying Weather-Cycling Patterns
'''

from __future__ import print_function
import sys,os
import warnings
warnings.filterwarnings('ignore')

import json
import random
import numpy as np
import pandas as pd
from time import time,clock,sleep
from itertools import combinations, permutations
from collections import Counter
from datetime import datetime as dtt
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,FormatStrFormatter
import matplotlib.patches as patches
import seaborn as sns

from DMA import dynamic_model_averaging as dma
from DMAu import dynamic_model_averaging_uov as dmau
from init import Model_Preparation as MdPp

font = {'family':'normal','size':18}
matplotlib.rc('font',**font)
sns.set(font_scale=2)
sns.set_style('ticks')

SEED = 0
np.random.seed(SEED)
random.seed(SEED)


if __name__ == '__main__':
	# ------------------------------ INITIALIZATION ------------------------------
	path = 'citibike/'
	out_path = 'data/' # '../data/','../mat','../functions'
	fig_path = 'fig/'  # 'fig/lag_', 'fig/', 'fig0/'

	# --------------------------- MODEL SPECIFICATION ----------------------------
	'''
	Select stationarity transformations (only for exogenous variables)
	1: use fully stationary variables
	2: use approximately stationary variables
	3: do not transform to stationarity
	'''
	stationarity = 2

	'''
	Select set of predictors to use
	1: use all predictors
	2: use only
	3: use only
	4: use no exogenous predictors (estimate AR(p) model)
	'''
	use_x = 3

	'''
	Select other exogenous predictors
	0: do not use other exogenous predictors
	1: use other exogenous predictors (age, gender etc.)
	'''
	use_other = 0

	'''
	How to treat missing values (only in the begining of the sample)
	1: fill in missing values with zeros, let the KF do the rest
	2: trim quarters which are not observed (that way we lose information
	    for variables which have no missing values)
	'''
	miss_treatment = 1

	'''
	Select a dataset
	1: daily data
	2: tourist season
	'''
	dataset = 1

	'''
	Estimate intercept?
	0: no
	1: yes
	'''
	intercept = 1

	'''
	Define lags
	plag: lags of dependent variables
	hlag: lags of exogenous variables
	'''
	plag = 0
	hlag = 0

	'''
	Where to apply DMA and DMS?
	1: only on the exogenous variables
	2: on the exogenous and the lags of the dependent (requires plag>0)
	3: on the exogenous, the lags of the dependent and the intercept (requires 'intercept=1')
	'''
	apply_dma = 1

	'''
	Whether to consider unknown observational variance?
	0: Normal assumption
	1: Student's t distribution
	'''
	unknown_var = 0

	'''
	Forgetting factors
	LAMBDA: for the time-varying parameters theta
	ALPHA: for the model switching
	KAPPA: for the error covariance matrix
	'''
	LAMBDA = 0.95
	ALPHA = 0.95
	KAPPA = 0.95

	'''
	Forgetting method on model switching probabilities
	1: linear forgetting
	2: exponential forgetting
	'''
	forgetting_method = 2

	'''
	Initial values on time-varying parameters
	theta[0] ~ N(PRM, PRV * I)
	1: diffuse N(0,4)
	2: data-based prior
	'''
	prior_theta = 2

	'''
	Initialize measurement error covariance V[t]
	1: a small positive value (but not exactly zero)
	2: a quarter of the variance of the initial data
	'''
	initial_V_0 = 2

	'''
	Initialize DMA weights
	1: equal weights
	'''
	initial_DMA_weights = 1

	'''
	Define expert opinion (prior) on model weight
	1: equal weights on all models
	2: no prior expert opinion
	'''
	expert_opinion = 2

	'''
	Define forecast horizon (applied to direct forecasts)
	'''
	h_fore = 1

	'''
	Define the last observation of the first sample
	used to start producing forecasts recursively.
	'''
	first_sample_ends = '2017-08-31'  # '2017-08-31', '2018-01-31'

	# MODEL SETTINGS
	## !! CHECK: dataset, unknown_var, hlag, LAMBDA, ALPHA, KAPPA, first_sample_ends, fig_path
	model = MdPp(path, out_path, stationarity, use_x, use_other,
				miss_treatment, dataset, intercept, plag, hlag,
				apply_dma, LAMBDA, ALPHA, KAPPA, forgetting_method,
				prior_theta, initial_V_0, initial_DMA_weights,
				expert_opinion, h_fore, first_sample_ends)

	'''
	Do a last check of the model specification inputs before you run the model.
	'''
	model.checkinput()

	# --------------------------------- DATA ----------------------------------
	T0 = time()
	# !! CHECK: dataX, dataY

	sheetX = 'all6'  # part6/all6/all6-1/all12/all12-1/allmax/allmin
	dataX = pd.read_excel(out_path + 'weather.xlsx', sheetname=sheetX, header=0)
	# # x = pd.read_excel(out_path + 'weather.xlsx', sheetname='all12', header=0)
	# # dataX['weekdays'] = x['weekdays']
	# # print(dataX.head())
	nameX = list(dataX.columns)

	sheetY = 'young'
	# 'trips1.xlsx'
	# young/middle/elderly/young1-4
	# 'trips2.xlsx'
	# ycus/ysub/mcus/msub/ecus/esub
	# ymale/yfemale/mmale/mfemale/emale/efemale
	# y1sub/y2sub/y3sub/y4sub/
	# y1cus/y2cus/y3cus/y4cus/
	# y1male/y2male/y3male/y4male/
	# y1female/y2female/y3female/y4female/
	# gd1cus/gd2cus/gd1sub/gd2sub
	dataY = pd.read_excel(out_path + 'trips1.xlsx', sheetname=sheetY, header=0)
	nameY = list(dataY.columns)

	if dataset == 1:
		time = pd.read_excel(out_path + 'time.xlsx', 'dates1')
	elif dataset == 2:
		time = pd.read_excel(out_path + 'time.xlsx', 'dates2')
	timelab = time

	dataX = dataX.values
	dataY = dataY.values
	dataY -= dataY.min()
	dataY = dataY/dataY.max()

	# nY = dataY.shape[1]
	## Potential direction: multiple dependent variables (use nY)
	## Potential Direction: stations/odpairs

	# ---------------------------- PRELIMINARIES -----------------------------
	## STEP1: DATA HANDLING
	# Load data, transform them accordingly and create dependent variable y_t (CitiBike Ridership),
	# and independent ones Z_t (unrestricted) and z_t (restricted)
	model.data_in(dataX, nameX, dataY, nameY, timelab, sheetX)

	# From all the independent variables (intercept, lags of y_t and exogenous),
	# create a vector of the names of the variables that are actually used to forecast, called 'Xnames'
	model.effective_names()


	## STEP2: DEFINE MODELS
	# Now get all possible model combination and create a variable indexing all those 2**N models
	# where N is the number of variables in z_t to be restricted, called 'index'
	model.model_index()


	## STEP3: PRIORS
	# For data-based priors, get the first sample in the recursive forecasting exercise
	start = np.where(time.time == first_sample_ends)[0][0]
	model.prior_hyper(start)


	## STEP4: MODELING
	if unknown_var:
		print('Run: dynamic_model_averaging_uov')
		theta_update_all, prob_pred, prob_update, y_t_pred_h, variance, y_t_DMA = dmau(
									model.K, model.T, T0, h_fore, model.V_0,
									forgetting_method, ALPHA,
									model.y_t, model.z_t, model.Z_t, model.index_z_t,
									model.expert_weight, model.prob_0_prmean, model.inv_lambda,
									model.theta_0_prmean, model.theta_0_prvar)
	else:
		print('Run: dynamic_model_averaging')
		theta_update_all, prob_pred, prob_update, y_t_pred_h, variance, y_t_DMA = dma(
									model.K, model.T, T0, h_fore, model.V_0,
									forgetting_method, ALPHA, KAPPA,
									model.y_t, model.z_t, model.Z_t, model.index_z_t,
									model.expert_weight, model.prob_0_prmean, model.inv_lambda,
									model.theta_0_prmean, model.theta_0_prvar)


	# STEP5: RESULTS
	# find out the best models
	max_prob = np.zeros(model.T)
	best_model = np.zeros(model.T)
	for ii in range(model.T):
		temp = prob_pred[ii].tolist()
		max_prob[ii], best_model[ii] = max(temp), temp.index(max(temp))


	# make the prediction based on the best model
	y_t_BEST = np.zeros(model.T)
	var_BEST = np.zeros(model.T)
	for ii in range(model.T):
		y_t_BEST[ii] = y_t_pred_h[best_model[ii]][ii]
		var_BEST[ii] = variance[best_model[ii]][ii]


	# The start of recursively forecasts
	ind = start
	xl = len(y_t_DMA) - ind

	# Time ticks
	if dataset == 1:
		ticks = ['Sep.', 'Oct.', 'Nov.', 'Dec.', 'Jan.', 'Feb.', 'Mar.', 'Apr.', 'May.', 'Jun.', 'Jul.', 'Aug.']
		ticks1 = ['Sep.', '', '', 'Dec.', '', '', 'Mar.', '', '', 'Jun.', '', '']
	elif dataset == 2:
		ticks = ['Feb.', 'Mar.', 'Apr.', 'May.', 'Jun.', 'Jul.', 'Aug.', 'Sep.', 'Oct.']
		ticks1 = ['', 'Mar.', '', '', 'Jun.', '', '', 'Sep.', '']

	# Expected size of main predictors
	rs, cs = prob_pred.shape
	ss = [len(model.index[ii]) for ii in range(cs)]


	# Size of the selected models
	Esize = [sum(ss * prob_pred[ii+ind,:]) for ii in range(1,xl)]
	Csize = [len(model.index[int(best_model[ii+ind])]) for ii in range(1,xl)]
	fig = plt.figure(figsize=(15,9))
	ax = fig.add_subplot(111)
	(markers, stemlines, baseline) = plt.stem(Csize, linefmt='brown', markerfmt='brown')
	plt.setp(markers, marker='o', alpha=0.6)
	plt.plot(Esize, linewidth=2.0)
	ax.grid(color='grey', linestyle='--', linewidth=1.5, alpha=0.9)
	ax.legend([r'$N_{DMA}$', r'$N_{DMS}$'], title='', frameon=False, fancybox=False)
	ax.set_xlim(0, xl)
	ax.set_ylim(1, len(model.Xnames)-1)
	ax.xaxis.set_major_locator(plt.MaxNLocator(len(ticks)+1))
	ax.set_xticklabels(ticks,rotation=45)
	ax.spines['bottom'].set_visible(True)
	ax.spines['left'].set_visible(True)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.set_xlabel('')
	ax.set_ylabel('')
	ax.set_title('')  # Expected number of influencing factors at the average level over time
	plt.tight_layout()
	plt.savefig(fig_path + 'size_' + sheetX + '.pdf')


	# Plot the probs of the k-th model
	k = 21
	# plt.plot(prob_update[:,k])
	# plt.show()
	# Present the regression coefficients whose variables are used in model k
	mname = [model.Xnames[ii] for ii in model.index[k]]
	print('\n', k, mname, '\n')


	# How the probabilities move at each point in time and which is the best model for each "t"
	fig = plt.figure(figsize=(15,9))
	ax = fig.add_subplot(111)
	(markers, stemlines, baseline) = plt.stem(best_model+1, markerfmt='o')
	plt.setp(markers, alpha=0.6)
	ax.spines['bottom'].set_visible(True)
	ax.spines['left'].set_visible(True)
	ax.spines['right'].set_visible(True)
	ax.spines['top'].set_visible(True)
	ax.grid(color='grey', linestyle='--', linewidth=1.0, alpha=0.9)
	ax.set_xlim(0, xl-1)
	ax.set_ylim(1, model.K)
	ax.xaxis.set_major_locator(plt.MaxNLocator(len(ticks)+1))
	ax.set_xticklabels(ticks, rotation=45)
	ax.set_xlabel('')
	ax.set_ylabel('')
	ax.set_title('')
	plt.tight_layout()
	plt.savefig(fig_path + 'one_' + sheetX + '.pdf')


	# Comparison between DMA predictions and Best model predictions based on the original observation
	fig = plt.figure(figsize=(15,9))
	ax = fig.add_subplot(111)
	color_sequence = ['#aec7e8', '#ff9896', '#9467bd']
	# color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
						# '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
						# '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
						# '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
	plt.plot(model.y_t[ind:-h_fore], linewidth=2.5, color=color_sequence[0])
	plt.plot(y_t_DMA[ind:], linewidth=2.5, color=color_sequence[1])
	plt.plot(y_t_BEST[ind:], linewidth=2.5, color=color_sequence[2])
	for pc in (1,3):
		ax.add_patch(patches.Rectangle((pc*91,0), 91, 1, alpha=0.15, facecolor='peachpuff'))
	ax.grid(color='grey', linestyle='--', linewidth=1.0, alpha=0.9)
	ax.legend([r'$y$', r'$y_{DMA}$', r'$y_{DMS}$'], title='', frameon=False, fancybox=False)
	ax.set_xlim(0, xl)
	ax.set_ylim(0, 1)
	ax.xaxis.set_major_locator(plt.MaxNLocator(len(ticks)+1))
	ax.set_xticklabels(ticks, rotation=45)
	ax.spines['bottom'].set_visible(True)
	ax.spines['left'].set_visible(True)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.set_xlabel('')
	ax.set_ylabel('')
	ax.set_title('')  # The original and estimated series (DMA/DMS)
	plt.tight_layout()
	plt.savefig(fig_path + 'predict_' + sheetX + '.pdf')


	# Maximum probability of a single model at each point in time
	fig = plt.figure(figsize=(12,9))
	ax = fig.add_subplot(111)
	plt.plot(max_prob[ind:], c='blue', linewidth=1.5)
	ax.grid(color='grey', linestyle='--', linewidth=1.0, alpha=0.8)
	ax.set_xlim(0, xl)
	# ax.set_ylim(1, model.K)
	ax.xaxis.set_major_locator(plt.MaxNLocator(len(ticks)))
	ax.set_xticklabels(ticks, rotation=0)
	ax.set_xlabel('')
	ax.set_ylabel('')
	ax.set_title('')  # Maximum probability of a single model over time
	plt.tight_layout()
	plt.savefig(fig_path + 'maxprob_' + sheetX + '.pdf')


	# The combination of prediction models
	table = Counter(best_model[ind:]).most_common(4)
	fig = plt.figure(figsize=(12,9))
	ax = fig.add_subplot(111)
	for ii,tb in enumerate(table):
		xid = model.index[int(tb[0])]
		xids = [model.Xnames[i] for i in xid]
		print(tb, xids)
		plt.plot(prob_update[:,int(tb[0])], linewidth=1.5)
	ax.grid(color='grey', linestyle='--', linewidth=1.0, alpha=0.6)
	ax.legend(['MODEL - ' + str(int(i[0])) for i in table], title='', frameon=False, fancybox=False)
	ax.xaxis.set_major_locator(plt.MaxNLocator(len(ticks)+1))
	ax.set_xticklabels(ticks, rotation=45)
	ax.spines['bottom'].set_visible(True)
	ax.spines['left'].set_visible(True)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.set_xlim(0, xl)
	# ax.set_ylim(0, 0.2)
	plt.axhline(y=0.1, color='black', linestyle='--', linewidth=1.0, alpha=0.6)
	plt.tight_layout()
	plt.savefig(fig_path + 'table_' + sheetX + '.pdf')


	# Forecast statistics
	# writer = pd.ExcelWriter(out_path + 'forecast.xlsx')  # Comment out
	MAFE_DMA = np.abs((model.y_t[h_fore:model.T] - y_t_DMA[:model.T-h_fore])/model.y_t[h_fore:model.T])
	MSFE_DMA = (model.y_t[h_fore:model.T] - y_t_DMA[:model.T-h_fore])**2

	MAFE_DMS = np.abs((model.y_t[h_fore:model.T] - y_t_BEST[:model.T-h_fore])/model.y_t[h_fore:model.T])
	MSFE_DMS = (model.y_t[h_fore:model.T] - y_t_BEST[:model.T-h_fore])**2

	# forecast = np.matrix([sum(MAFE_DMA[xl:]), np.sqrt(sum(MSFE_DMA[xl:])), sum(MAFE_DMS[xl:]), np.sqrt(sum(MSFE_DMS[xl:]))])
	forecast = np.matrix([np.mean(MAFE_DMA[xl:]), np.sqrt(np.mean(MSFE_DMA[xl:])), 
								np.mean(MAFE_DMS[xl:]), np.sqrt(np.mean(MSFE_DMS[xl:]))])
	forecast = pd.DataFrame(forecast, columns=['MAFE_DMA', 'MSFE_DMA', 'MAFE_DMS', 'MSFE_DMS'], index=[str(hlag)+sheetX+sheetY])
	print(forecast)
	# wb = openpyxl.load_workbook(out_path + 'forecast.xlsx')
	# ws = wb.create_sheet(title=str(hlag)+sheetX+sheetY, index=0)
	# rows = dataframe_to_rows(forecast)
	# for r_idx, row in enumerate(rows,1):
	# 	for c_idx, value in enumerate(row,1):
	# 		ws.cell(row=r_idx, column=c_idx, value=value)
	# wb.save(out_path + 'forecast.xlsx')
	# # forecast.to_excel(out_path + 'forecast.xlsx', sheet_name=sheetX + sheetY)
	# # writer.save()  # Comment out


	# Make probability plots of restricted variables
	if len(model.index[model.K-1]) == 0:
		firstX = 0
		lastX = model.z_t.shape[1]
	else:
		firstX = max(model.index[model.K-1]) + 1
		lastX = max(model.index[model.K-1]) + model.z_t.shape[1] + 1

	if lastX-firstX == 6:
		fig = plt.figure(figsize=(18,11))
	elif lastX-firstX == 12:
		fig = plt.figure(figsize=(18,22))
	iplot = 1
	backup = np.zeros((xl, lastX-firstX))
	# writer = pd.ExcelWriter(out_path + 'prob_variable.xlsx')  # Comment out

	g_index_tmp = []
	for index_variable in range(firstX, lastX):
		g_index = []
		for ii in range(model.K):
			ddd = len(np.where(np.array(model.index[ii]) == index_variable)[0])
			if ddd:
				g_index.append(ii)
		g_index_tmp.append(g_index)

	for index_variable,g_index in enumerate(g_index_tmp):
		ax = fig.add_subplot(int((lastX-firstX)/3), 3, iplot)
		prob_variable = np.sum(np.squeeze(prob_update[:,g_index]), axis=1)
		plt.plot(prob_variable[ind:], alpha=0.9, linewidth=2.0, color='royalblue')
		backup[:,iplot-1] = prob_variable[ind:]
		iplot += 1

		if dataset == 1:
			for pc in (1,3):
				ax.add_patch(patches.Rectangle((pc*91,0), 91, 1, alpha=0.15, facecolor='peachpuff'))
		elif dataset == 2:
			ax.add_patch(patches.Rectangle((0,0), 31, 1, alpha=0.15, facecolor='peachpuff'))
			ax.add_patch(patches.Rectangle((121,0), 91, 1, alpha=0.15, facecolor='peachpuff'))

		ax.grid(color='grey', linestyle='--', linewidth=1.0, alpha=0.25)
		# ax.legend()
		ax.set_xlim(0,xl)
		ax.set_ylim(0,1)
		ax.xaxis.set_major_locator(plt.MaxNLocator(len(ticks1)+1))
		ax.set_xticklabels(ticks1, rotation=45)
		ax.spines['bottom'].set_visible(True)
		ax.spines['left'].set_visible(True)
		ax.spines['bottom'].set_color('grey')
		ax.spines['left'].set_color('grey')
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.set_xlabel('')
		ax.set_ylabel('')
		ax.set_title(r'$\rm %s$' % model.Xnames[index_variable+1])  # Time-varying probability of inclusion of variable xxx

	plt.tight_layout()
	plt.savefig(fig_path + 'probs_' + sheetX + '.pdf')
	backup = pd.DataFrame(backup, columns=model.Xnames[firstX:lastX])

	# wb = openpyxl.load_workbook(out_path + 'prob_variable.xlsx')
	# ws = wb.create_sheet(title=str(hlag)+sheetX+sheetY, index=0)
	# rows = dataframe_to_rows(backup)
	# for r_idx, row in enumerate(rows,1):
	# 	for c_idx, value in enumerate(row,1):
	# 		ws.cell(row=r_idx, column=c_idx, value=value)
	# wb.save(out_path + 'prob_variable.xlsx')
	# # backup.to_excel(out_path + 'prob_variable.xlsx', sheet_name=sheetX+sheetY)
	# # writer.save()  # Comment out


	# Cross comparisons
	subs = [['0all6young', '0all6middle', '0all6elderly'], ['0part6gd1sub', '0part6gd2sub', '0part6gd1cus', '0part6gd2cus']]
	legends = [['young', 'middle', 'elderly'], ['sub.male', 'sub.female', 'cus.male', 'cus.female']]

	for v,sub in enumerate(subs):
		subset = []
		for ii in sub:
			temp = pd.read_excel('data/prob_variable.xlsx', sheet_name=ii, index_col=0, header=0)
			subset.append(np.array(temp))
		print(len(subset), subset[0].shape)

		fig = plt.figure(figsize=(18,11))
		for index_variable,g_index in enumerate(g_index_tmp):
			ax = fig.add_subplot(2,3,index_variable+1)

			if len(subset) == 3:
				colors = ['royalblue', 'brown', 'orange']
				for idx in range(len(subset)):
					plt.plot(subset[idx][:,index_variable], alpha=0.7, linewidth=2.0, c=colors[idx])

				ax.set_xlim(0, 396-ind)
				ticks1 = ['Sep.', '', '', 'Dec.', '', '', 'Mar.', '', '', 'Jun.', '', '']
				for pc in (1,3):
					ax.add_patch(patches.Rectangle((pc*91,0), 91, 1, alpha=0.15, facecolor='peachpuff'))

			elif len(subset) == 4:
				plt.plot(subset[0][:,index_variable], c='royalblue', alpha=0.8, linestyle='-', linewidth=2.0)
				plt.plot(subset[1][:,index_variable], c='royalblue', alpha=0.8, linestyle=':', linewidth=2.0)

				plt.plot(subset[2][:,index_variable], c='brown', alpha=0.8, linestyle='-', linewidth=2.0)
				plt.plot(subset[3][:,index_variable], c='brown', alpha=0.8, linestyle=':', linewidth=2.0)

				ax.fill_between(list(range(304-ind-1)), 
					subset[0][:,index_variable], subset[1][:,index_variable], 
					facecolor='aqua', alpha=0.29)

				ax.fill_between(list(range(304-ind-1)), 
					subset[2][:,index_variable], subset[3][:,index_variable], 
					facecolor='yellow', alpha=0.29)

				ax.set_xlim(0, 304-ind)
				ticks1 = ['', 'Mar.', '', '', 'Jun.', '', '', 'Sep.', '']
				ax.add_patch(patches.Rectangle((0,0), 31, 1, alpha=0.15, facecolor='peachpuff'))
				ax.add_patch(patches.Rectangle((121,0), 91, 1, alpha=0.15, facecolor='peachpuff'))

			ax.grid(color='grey', linestyle='--', linewidth=1.0, alpha=0.25)
			ax.set_ylim(0,1)
			ax.xaxis.set_major_locator(plt.MaxNLocator(len(ticks1)+1))
			ax.set_xticklabels(ticks1, rotation=45)
			ax.set_title(r'$\rm %s$' % model.Xnames[index_variable+1])

			ax.spines['bottom'].set_visible(True)
			ax.spines['left'].set_visible(True)
			ax.spines['bottom'].set_color('grey')
			ax.spines['left'].set_color('grey')
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)

			if index_variable == 3:
				ax.legend(legends[v], title='', frameon=False, fancybox=False)

		ax.set_xlabel('')
		ax.set_ylabel('')
		plt.tight_layout()
		plt.savefig(fig_path + 'cross_' + str(len(subset)) + '.pdf')


	# Posterior coefficients
	time_range = len(theta_update_all[0])
	model_size = len(theta_update_all)
	theta_set = []

	for t_1 in range(time_range):
		temp = np.zeros((model_size, lastX-firstX+1))
		for ii in range(model.K):
			temp[ii, np.array(model.index[ii])] = theta_update_all[ii][t_1].T
		theta_set.append(temp[:,1:])

	theta_mean = np.zeros((time_range,lastX-firstX))
	theta_max = np.zeros((time_range,lastX-firstX))
	theta_min = np.zeros((time_range,lastX-firstX))
	for index_variable,g_index in enumerate(g_index_tmp):
		for t_2 in range(time_range):
			theta_mean[t_2,index_variable] = np.mean(theta_set[t_2][g_index,index_variable])
			theta_max[t_2,index_variable] = np.max(theta_set[t_2][g_index,index_variable])
			theta_min[t_2,index_variable] = np.min(theta_set[t_2][g_index,index_variable])

	# theta_best1 = np.zeros((time_range,lastX-firstX))
	# for index_variable,g_index in enumerate(g_index_tmp):
	# 	for t_2 in range(time_range):
	# 		theta_best1[t_2,index_variable] = theta_set[t_2][1,index_variable]  # 13, 37

	fig = plt.figure(figsize=(18,11))
	for ii in range(lastX-firstX):
		ax = fig.add_subplot(2,3,ii+1)

		ax.fill_between(list(range(time_range-ind)), 
					theta_max[ind:,ii], theta_min[ind:,ii],
					facecolor='brown', alpha=0.22)

		plt.plot(theta_mean[ind:,ii], alpha=0.91, linewidth=2.0, color='royalblue')

		ax.set_xlim(0,xl)
		xmin, xmax, ymin, ymax = ax.axis()
		for pc in (1,3):
			ax.add_patch(patches.Rectangle((pc*91,ymin), 91, ymax-ymin, alpha=0.15, facecolor='peachpuff'))
		ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

		ax.grid(color='grey', linestyle='--', linewidth=1.0, alpha=0.25)
		ticks1 = ['Sep.', '', '', 'Dec.', '', '', 'Mar.', '', '', 'Jun.', '', '']
		ax.xaxis.set_major_locator(plt.MaxNLocator(len(ticks1)+1))
		ax.set_xticklabels(ticks1, rotation=45)

		ax.spines['bottom'].set_visible(True)
		ax.spines['left'].set_visible(True)
		ax.spines['bottom'].set_color('grey')
		ax.spines['left'].set_color('grey')
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)

		ax.set_xlabel('')
		ax.set_ylabel('')
		ax.set_title(r'$\rm %s$' % model.Xnames[ii+1])

	plt.tight_layout()
	plt.savefig(fig_path + 'theta_' + sheetX + '.pdf')

