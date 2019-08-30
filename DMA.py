# coding=utf-8
'''
The codes of the original DMA and DMS in MATLAB version are provided by:
"Koop, G., & Korobilis, D. (2012). Forecasting inflation using dynamic model averaging.
International Economic Review, 53(3), 867-886."
website: https://sites.google.com/site/dimitriskorobilis/matlab/dma
'''

from __future__ import print_function
import sys,os
import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np
import pandas as pd
from time import time,clock,sleep
from itertools import combinations, permutations
from collections import Counter,defaultdict
from datetime import datetime as dtt

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as patches
import seaborn as sns

font = {'family':'normal','size':17}
matplotlib.rc('font',**font)
sns.set(font_scale=1.8)
sns.set_style('ticks')

SEED = 0
np.random.seed(SEED)
random.seed(SEED)


# DYNAMIC MODEL AVERAGING
def dynamic_model_averaging(K, T, T0, h_fore, V_0,
							forgetting_method, ALPHA, KAPPA,
							y_t, z_t, Z_t, index_z_t,
							expert_weight, prob_0_prmean, inv_lambda,
							theta_0_prmean, theta_0_prvar):
	# initialize matrices
	theta_pred = defaultdict(list)
	R_t = defaultdict()

	prob_pred = np.zeros((T,K))
	y_t_pred = defaultdict(list)
	y_t_pred_h = defaultdict(list)

	e_t = defaultdict(list)
	A_t = np.zeros(K)
	V_t = defaultdict(list)

	theta_update = defaultdict(list)
	theta_update_all = defaultdict(list)
	S_t = defaultdict()
	variance = defaultdict(list)
	w_t = defaultdict(list)
	log_PL = np.zeros(K)
	prob_update = np.zeros((T,K))

	y_t_DMA = np.zeros(T)
	var_DMA = np.zeros(T)
	y_t_BEST = np.zeros(T)

	log_PL_DMA = np.zeros(T)
	log_PL_BEST = np.zeros(T)
	offset = 1e-20  ## for numerical stability

	# 1:T time periods
	for irep in range(T):
		if irep and irep % 50 == 0:
			print ('%d completed, %ds used.' % (irep, round(time()-T0, 4)))

		# get the sum of all K model probabilities
		if irep > 0:
			if forgetting_method == 1:
				# linear forgetting
				# this is the sum of the K model probabilities (all in multiplied by the forgetting factor)
				sum_prob = np.sum(ALPHA*prob_update[irep-1,:] + (1-ALPHA)*expert_weight)  # sum(,2)

			elif forgetting_method == 2:
				# exponential forgetting
				# this is the sum of the K model probabilities (all in the power of the forgetting factor)
				sum_prob_a = np.sum(np.multiply(prob_update[irep-1,:]**ALPHA, expert_weight**(1-ALPHA)))  # sum(,2)

		# reset log_PL, A_t, and R_t, to zero at each iteration
		log_PL = np.zeros(K)
		A_t = np.zeros(K)
		R_t = defaultdict()

		# 1:K competing models
		for k in range(K):
			x_t = np.c_[Z_t, z_t[:,index_z_t[k]]]

			# PREDICT
			if irep == 0:
				# predict theta[t] of Eq.()
				theta_pred[k] = np.asmatrix(theta_0_prmean[k]).T

				# predict R[t] of Eq.()
				R_t[k] = inv_lambda * theta_0_prvar[k]
				temp1 = prob_0_prmean**ALPHA

				# predict model probability of Eq.()
				prob_pred[irep,k] = temp1/(K*temp1)

			else:
				# predict theta[t] of Eq.()
				theta_pred[k] = theta_update[k]

				# predict R[t] of Eq.()
				R_t[k] = inv_lambda * S_t[k]

				if forgetting_method == 1:
					# linear forgetting
					prob_pred[irep,k] = (ALPHA*prob_update[irep-1,k] + (1-ALPHA)*expert_weight) / sum_prob

				elif forgetting_method == 2:
					# exponential forgetting
					# predict model probability of Eq.()
					prob_pred[irep,k] = (prob_update[irep-1,k]**ALPHA * expert_weight**(1-ALPHA) + offset) / (sum_prob_a+offset)

			# Implement individual-model predictions of the variable of interest
			# one step ahead prediction
			y_t_pred[k].append(np.matrix(x_t[irep,:]) * theta_pred[k]) ##dict

			# Do h_fore-step ahead prediction
			# predict t+h given t
			y_t_pred_h[k].append(np.matrix(x_t[irep+h_fore,:]) * theta_pred[k])

			# UPDATE
			# one-step ahead prediction error
			e_t[k].append(y_t[irep] - y_t_pred[k][irep])

			# some products of matrices, define for computational efficiency
			R_mat = R_t[k]
			xRx2 = np.dot(np.dot(x_t[irep,:], R_mat), x_t[irep,:])

			# Update V_t-measurement error covariance matrix using rolling moments estimator
			if irep == 0:
				V_t[k].append(V_0)
			else:
				A_t[k] = e_t[k][irep-1]**2
				V_t[k].append(KAPPA * V_t[k][irep-1] + (1-KAPPA) * A_t[k])

			# Update regression coefficient theta[t] and its covariance matrix S[t]
			Rx = np.dot(R_mat, np.asmatrix(x_t[irep,:]).T)
			KV = V_t[k][irep] + xRx2
			KG = Rx/KV
			theta_update[k] = theta_pred[k] + KG * e_t[k][irep]
			S_t[k] = R_mat - np.dot(KG, np.asmatrix(np.dot(x_t[irep,:], R_mat)))

			# save time-varying coeficients
			theta_update_all[k].append(theta_update[k])

			# Update model probability
			# Feed in the forecast mean and forecast variance 
			# and evaluate at the future dependent value a Normal density.
			# This density is called the predictive likelihood or posterior marginal likelihood
			# use f_l to update model weight or probability called w_t

			# the forecast variance of each model
			variance[k].append(V_t[k][irep] + xRx2)

			# when the x[t]*R[t]*x[t].T quantity is negative
			variance[k][irep] = np.abs(variance[k][irep])

			# the forecast mean
			## f_l = np.normpdf(y_t[irep,:], mean, np.sqrt(variance))
			mean = np.dot(x_t[irep,:], theta_pred[k])

			f_l = (1 / np.sqrt(2 * np.pi * variance[k][irep])) * np.exp(-0.5 * ((y_t[irep] - mean)**2 / variance[k][irep]))
			w_t[k].append(prob_pred[irep,k] * f_l)

			# Calculate log predictive likelihood for each model
			log_PL[k] = np.log(f_l + offset)
			# end cycling through all possible K models

		# calculate the denominator of Eq.() (the sum of the w.T*s)
		sum_w_t = 0
		for k_2 in range(K):
			sum_w_t += w_t[k_2][irep]

		# calculate the updated model probabilities in Eq.()
		for k_3 in range(K):
			prob_update[irep,k_3] = (w_t[k_3][irep] + offset) / (sum_w_t + offset)

		# Do DMA forecasting with the predictions for each model and the associated model probabilities.
		for k_4 in range(K):
			model_i_weight = prob_pred[irep,k_4]

			# temp_XXXs calculate individual model quantities, weighted by model probabilities
			temp_pred = y_t_pred_h[k_4][irep] * model_i_weight
			temp_var = variance[k_4][irep] * model_i_weight
			temp_logPL = log_PL[k_4] * model_i_weight

			# the mean DMA forecast
			y_t_DMA[irep] = y_t_DMA[irep] + temp_pred

			# the variance of the DMA forecast
			var_DMA[irep] = var_DMA[irep] + temp_var

			# the DMA predictive likelihood
			log_PL_DMA[irep] = log_PL_DMA[irep] + temp_logPL

		# get log_PL_BEST here
		prob_update_list = prob_update[irep,:].tolist()
		temp_best_model = prob_update_list.index(max(prob_update_list))
		log_PL_BEST[irep] = log_PL[temp_best_model]

	return theta_update_all, prob_pred, prob_update, y_t_pred_h, variance, y_t_DMA

