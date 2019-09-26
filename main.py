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

from scipy import stats
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


def forecast_metrics(model, h_fore, y_t_DMA, y_t_BEST):
	'''
	Model forecast comparisons
	'''
	MAFE_DMA = np.abs((model.y_t[h_fore:model.T] - y_t_DMA[:model.T-h_fore])/model.y_t[h_fore:model.T])
	MSFE_DMA = (model.y_t[h_fore:model.T] - y_t_DMA[:model.T-h_fore])**2

	MAFE_DMS = np.abs((model.y_t[h_fore:model.T] - y_t_BEST[:model.T-h_fore])/model.y_t[h_fore:model.T])
	MSFE_DMS = (model.y_t[h_fore:model.T] - y_t_BEST[:model.T-h_fore])**2

	forecast = np.matrix([np.mean(MAFE_DMA[xl:]), np.sqrt(np.mean(MSFE_DMA[xl:])), 
								np.mean(MAFE_DMS[xl:]), np.sqrt(np.mean(MSFE_DMS[xl:]))])
	forecast = pd.DataFrame(forecast, columns=['MAFE_DMA', 'MSFE_DMA', 'MAFE_DMS', 'MSFE_DMS'], index=[str(hlag)+sheetX+sheetY])
	print('\n',forecast)

	eps_a = model.y_t[h_fore:model.T] - y_t_DMA[:model.T-h_fore]
	eps_s = model.y_t[h_fore:model.T] - y_t_BEST[:model.T-h_fore]

	distr = np.round(np.matrix([stats.skew(eps_a), stats.skew(eps_s), 
					stats.kurtosis(eps_a,fisher=False), stats.kurtosis(eps_s,fisher=False)]), 6)
	distr = pd.DataFrame(distr, columns=['skew_a', 'skew_s', 'kurtosis_a', 'kurtosis_s'])
	print(distr,'\n')


def best_predictive_models(model, best_model):
	'''
	The combination of predictive models
	'''
	table = Counter(best_model[ind:]).most_common(10)
	for ii,tb in enumerate(table):
		xid = model.index[int(tb[0])]
		xids = [model.Xnames[i] for i in xid]
		print(tb, xids)


def best_model_size(model, prob_pred, best_model, ticks, xl, ind, fig_path, sheetX):
	'''
	Expected number of main influencing factors at the average level over time
	'''
	rs, cs = prob_pred.shape
	ss = [len(model.index[ii]) for ii in range(cs)]

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
	ax.set_title('')
	plt.tight_layout()
	plt.savefig(fig_path + 'size_' + sheetX + '.pdf')


def best_model_index(model, best_model, ticks, xl, fig_path, sheetX):
	'''
	How the probabilities move at each point in time and which is the best model for each "t"
	'''
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


def posterior_inclusion_probability(firstX, lastX, g_index_tmp, dataset, model, prob_update, ticks1, xl, fig_path, sheetX):
	'''
	Time-varying posterior inclusion probability of restricted variables
	'''
	if lastX-firstX == 6:
		fig = plt.figure(figsize=(18,11))
	elif lastX-firstX == 12:
		fig = plt.figure(figsize=(18,22))
	iplot = 1

	for index_variable,g_index in enumerate(g_index_tmp):
		ax = fig.add_subplot(int((lastX-firstX)/3), 3, iplot)
		prob_variable = np.sum(np.squeeze(prob_update[:,g_index]), axis=1)
		plt.plot(prob_variable[ind:], alpha=0.9, linewidth=2.0, color='royalblue')
		iplot += 1

		if dataset == 1:
			for pc in (1,3):
				ax.add_patch(patches.Rectangle((pc*91,0), 91, 1, alpha=0.15, facecolor='peachpuff'))
		elif dataset == 2:
			ax.add_patch(patches.Rectangle((0,0), 31, 1, alpha=0.15, facecolor='peachpuff'))
			ax.add_patch(patches.Rectangle((121,0), 91, 1, alpha=0.15, facecolor='peachpuff'))

		ax.grid(color='grey', linestyle='--', linewidth=1.0, alpha=0.25)
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
		ax.set_title(r'$\rm %s$' % model.Xnames[index_variable+1])

	plt.tight_layout()
	plt.savefig(fig_path + 'probs_' + sheetX + '.pdf')


def posterior_means_coefficients(firstX, lastX, g_index_tmp, model, theta_update_all, ticks1, xl, ind, fig_path, sheetX):
	'''
	Posterior means of regression coefficients of restricted variables
	'''
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



if __name__ == '__main__':
	# ------------------------------ INITIALIZATION ------------------------------
	path = 'citibike/'
	out_path = 'data/'
	fig_path = 'fig/'

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

	first_sample_ends = '2017-08-31'


	# MODEL SETTINGS
	print('dataset: %s\nunknown_var: %s\nhlag: %s\nLAMBDA: %.2f\nALPHA: %.2f\nKAPPA: %.2f\nfirst_sample_ends: %s\nfig_path: %s\n'
		% (dataset, unknown_var, hlag, LAMBDA, ALPHA, KAPPA, first_sample_ends, fig_path))

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
	sheetX = 'all6'
	dataX = pd.read_excel(out_path + 'weather.xlsx', sheetname=sheetX, header=0)
	nameX = list(dataX.columns)

	sheetY = 'young'
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
	print('dataX:', sheetX, dataX.shape, '\ndataY:', sheetY, dataY.shape, '\n')


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


	# Obtain the first and the last variable index
	if len(model.index[model.K-1]) == 0:
		firstX = 0
		lastX = model.z_t.shape[1]
	else:
		firstX = max(model.index[model.K-1]) + 1
		lastX = max(model.index[model.K-1]) + model.z_t.shape[1] + 1


	# Obtain the model index with the specific factor
	g_index_tmp = []
	for index_variable in range(firstX, lastX):
		g_index = []
		for ii in range(model.K):
			ddd = len(np.where(np.array(model.index[ii]) == index_variable)[0])
			if ddd:
				g_index.append(ii)
		g_index_tmp.append(g_index)


	# Output results
	forecast_metrics(model, h_fore, y_t_DMA, y_t_BEST)

	best_predictive_models(model, best_model)

	best_model_size(model, prob_pred, best_model, ticks, xl, ind, fig_path, sheetX)

	best_model_index(model, best_model, ticks, xl, fig_path, sheetX)

	posterior_inclusion_probability(firstX, lastX, g_index_tmp, dataset, model, prob_update, ticks1, xl, fig_path, sheetX)

	posterior_means_coefficients(firstX, lastX, g_index_tmp, model, theta_update_all, ticks1, xl, ind, fig_path, sheetX)

