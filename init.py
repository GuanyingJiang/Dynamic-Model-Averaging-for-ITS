# coding=utf-8
from __future__ import print_function
import sys,os
import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np
import pandas as pd
from time import time, clock, sleep
from itertools import combinations, permutations
from collections import Counter, defaultdict
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


# MODEL INITIALIZATION
class Model_Preparation(object):
	def __init__(self, path, out_path, stationarity, use_x, use_other,
				miss_treatment, dataset, intercept, plag, hlag,
				apply_dma, LAMBDA, ALPHA, KAPPA, forgetting_method,
				prior_theta, initial_V_0, initial_DMA_weights,
				expert_opinion, h_fore, first_sample_ends):
		super(Model_Preparation, self).__init__()

		self.path, self.out_path, self.dataset = path, out_path, dataset
		self.use_x, self.use_other, self.intercept = use_x, use_other, intercept
		self.plag, self.hlag = plag, hlag
		self.LAMBDA, self.ALPHA, self.KAPPA = LAMBDA, ALPHA, KAPPA
		self.stationarity = stationarity
		self.miss_treatment = miss_treatment
		self.apply_dma = apply_dma
		self.forgetting_method = forgetting_method
		self.prior_theta = prior_theta
		self.initial_V_0, self.initial_DMA_weights = initial_V_0, initial_DMA_weights
		self.expert_opinion = expert_opinion
		self.h_fore, self.first_sample_ends = h_fore, first_sample_ends		

	def checkinput(self):
		assert (self.intercept in (0,1)), 'Wrong specification of the intercept. Please choose 0 or 1.'
		assert (self.plag >= 0 and isinstance(self.plag, int)), 'Lag-length of the dependent variable (plag) must be a positive integer.'
		assert (self.hlag >= 0 and isinstance(self.hlag, int)), 'Lag-length of the exogenous variables (hlag) must be a positive integer.'

		assert (self.apply_dma in (1,2,3)), 'Wrong specification of apply_dma. Please choose 1, 2, or, 3.'

		if self.apply_dma == 3 and self.intercept != 1:
			raise ValueError('Cannot set apply_dma = 3, when no intercept is defined in the model.')
		if self.apply_dma == 2 and self.plag == 0:
			raise ValueError('Cannot set apply_dma = 2, when no lags are defined in the model.')

		assert (0 <= self.LAMBDA <= 1), 'The forgetting factors must take values between 0 and 1.'
		assert (0 <= self.ALPHA <= 1), 'The forgetting factors must take values between 0 and 1.'
		assert (0 <= self.KAPPA <= 1), 'The forgetting factors must take values between 0 and 1.'

		assert (self.forgetting_method in (1,2)), 'The forgetting method must be 1 or 2.'
		assert (self.prior_theta in (1,2)), 'Wrong specification of prior_theta. Please choose 1 or 2.'

		assert (self.initial_V_0 in (1,2)), 'Wrong initialization of V_0. Please choose 1 or 2.'
		assert (self.initial_DMA_weights == 1), 'Wrong initialization DMA weights.'

		assert (self.expert_opinion in (1,2)), 'Wrong initialization of expert opinion. Please choose 1 or 2.'
		assert (self.h_fore > 0), 'Wrong selection of forecast horizon.'

		return self

	@staticmethod
	def mlag2(X, p):
		Traw,N = X.shape
		Xlag = np.zeros((Traw, N*p))
		for ii in range(p):
			Xlag[p:Traw, N*ii:N*(ii+1)] = X[(p-ii-1):(Traw-ii-1), :N]
		return Xlag

	@staticmethod
	def stationarity(x, types):
		nr,nc = x.shape
		txx = np.zeros((nr,nc))
		adf = np.zeros((3,nc))

		for ii in range(nc):
			tx = np.zeros(nr)

			if types[ii] == 1:
				tx = x[:,ii]  #original
			elif types[ii] ==2:
				tx[1:] = np.diff(x[:,ii])  #first-order difference
			elif types[ii] == 3:
				tx = np.log(x[:,ii])  #logarithm
			elif types[ii] == 4:
				tx[1:] = np.diff(np.log(x[:,ii]))  #first-order logarithmic difference

			txx[:,ii] = tx
			# lg, pvalue, stats = adftest(txx[:,ii],0.95)  # ALTER
			# adf[:,ii] = np.array([lg, pvalue, stats])  # ALTER
		return txx

	def data_in(self, dataX, nameX, dataY, nameY, timelab, sheetX):
		# Import and preprocess data
		if self.dataset == 1:
			if sheetX == 'allmin':
				types = [1,2,1,1,1,2]
			elif sheetX == 'all12':
				types = [2,2,2,1,1,1,1,1,2,2,1,1]
			elif sheetX == 'all12-1':
				types = [2,2,2,1,1,1,1,1,2,2,1]
			elif sheetX == 'all6-1':
				types = [1,2,2,1,1,2,1]
			else:
				types = [1,2,2,1,1,2] #all6/allmax/lag
		elif self.dataset == 2:
			types = [1,2,2,2,1,2]

		dataX = Model_Preparation.stationarity(dataX, types)
		# dataX -= np.min(dataX,0)
		# dataX /= np.max(dataX,0)

		ncol = dataX.shape[1]
		Xindex = list(range(ncol))
		namesX = [nameX[i] for i in Xindex]

		Yindex = 0
		namesY = nameY[Yindex]

		X = dataX[:,Xindex]
		Y = dataY[:,Yindex]
		T = Y.shape[0]  #the number of cases
		h = X.shape[1]  #the number of predictors

		LAGS = max(self.plag, self.hlag)  #the number of lags
		if self.plag >0:
			ylag = Y
			if self.plag > 1:
				ylag = np.c_[ylag, Model_Preparation.mlag2(Y, self.plag-1)]
				ylag = ylag[LAGS:,:]
		else:
			ylag = []
		xlag = Model_Preparation.mlag2(X,self.hlag)
		xlag = xlag[LAGS:,:]

		if self.apply_dma == 1:
			z_t = np.c_[X[LAGS:T,:], xlag]
			if self.intercept == 1:
				try:
					Z_t = np.c_[np.ones((T-LAGS,1)), ylag]  ## if ylag!=[]
				except:
					Z_t = np.ones((T-LAGS,1))
			elif self.intercept == 0:
				Z_t = ylag
		elif self.apply_dma == 2:
			z_t = np.c_[ylag, X[LAGS:T,:], xlag]
			if self.intercept == 1:
				Z_t = np.ones((T-LAGS,1))
			elif self.intercept == 0:
				Z_t = []
		elif self.apply_dma == 3:
			z_t = np.c_[np.ones((T-LAGS,1)), ylag, X[LAGS:T,:], xlag]
			Z_t = []

		y_t = Y[LAGS:]  # shape(396,)
		timelab = timelab[LAGS:]
		timelab = timelab[:-self.h_fore]

		T = y_t.shape[0]
		T -= self.h_fore

		# states
		self.namesX = namesX
		self.z_t, self.Z_t, self.y_t = z_t, Z_t, y_t
		self.T, self.h, self.timelab = T, h, timelab
		return self

	def effective_names(self):
		# create vector with names of variables
		namesARY = ['']*(self.plag)
		for ii in range(self.plag):
			namesARY[ii] = 'ARY_{t-' + str(self.h_fore+ii) + '}'

		namesARX = ['']*(self.h*(self.hlag+1))  ##h-the number of predictors
		for j in range(self.h):
			namesARX[j] = self.namesX[j] + '_t'  ##namesX-the names of predictors

		if self.hlag >= 1:
			for ii in range(self.hlag):
				for j in range(self.h):
					namesARX[(ii+1)*self.h+j] = self.namesX[j] + '_{t-' + str(self.h_fore+ii) + '}'

		if self.intercept == 1:
			Xnames = ['intercept'] + namesARY + namesARX
		elif self.intercept == 0:
			Xnames = namesARY + namesARX

		self.Xnames = Xnames
		return self

	@staticmethod
	def combntns(choicevec, choose):
		return list(combinations(choicevec,choose))

	def model_index(self):
		## Form all possible model combinations
		# If z_t (the full model matrix of regressors) has N elements,
		# all possible combinations are (2**N-1), i.e. 2**N minus the model
		# with all predictors/constant excluded (y_t = error)
		N = self.z_t.shape[1]
		comb = []
		for nn in range(N):
			temp = Model_Preparation.combntns(range(N), nn+1)
			comb.extend(temp)
		comb.append(())

		# create a variable indexing all possible models using only z_t
		index_z_t = comb

		# total number of models
		K = np.round(2**N)

		# create a big index variable for all possible model combinations using x_t
		# which contains both the unrestricted and restricted variables
		index = []
		if self.apply_dma == 1:
			if self.intercept == 1:
				for iii in range(K-1):
					index.append(tuple(range(self.plag +1)) + tuple([i + self.plag +1 for i in index_z_t[iii]]))
			elif self.intercept == 0:
				for iii in range(K-1):
					index.append(tuple(range(self.plag)) + tuple([i + self.plag for i in index_z_t[iii]]))
		elif self.apply_dma == 2:
			if self.intercept == 1:
				for iii in range(K-1):
					index.append(tuple([i+1 for i in (0,)+index_z_t[iii]]))
			elif self.intercept == 0:
				index = index_z_t
		elif self.apply_dma == 3:
			index = index_z_t
		if len(index) < K:
			index.append((0,))

		# states
		self.K, self.index, self.index_z_t = K, index, index_z_t
		return self

	def prior_hyper(self, start):
		# Set prior means and variances and initial conditions
		# theta_0 ~ N(theta_0_prmean,theta_0_prvar)
		theta_0_prmean = []
		theta_0_prvar = []

		# diffuse
		if self.prior_theta == 1:
			for ll in range(self.K):
				theta_0_prmean.append(np.zeros(len(self.index[ll])))
				theta_0_prvar.append(np.eye(len(self.index[ll])) *4)

		# data-based
		elif self.prior_theta == 2:
			for ll in range(self.K):
				if len(self.Z_t) == 0:
					x_t = self.z_t[:start, self.index_z_t[ll]]
				else:
					x_t = np.c_[self.Z_t[:start,:], self.z_t[:start, self.index_z_t[ll]]]

				varx = np.var(x_t, ddof=1, axis=0)
				varx[varx==0]=0.01
				vary = np.var(self.y_t[:start])
				theta_0_prmean.append(np.zeros(len(self.index[ll])))
				theta_0_prvar.append(np.diag(vary/varx).T *2)

		self.theta_0_prmean = theta_0_prmean
		self.theta_0_prvar = theta_0_prvar

		# Initial value of measurement error covariance matrix V_t
		if self.initial_V_0 == 1:
			V_0 = 1e-3
		elif self.initial_V_0 == 2:
			V_0 = np.var(self.y_t[:start])/4
		self.V_0 = V_0

		# Initial model probability for each individual model
		if self.initial_DMA_weights == 1:
			prob_0_prmean = 1/self.K
		self.prob_0_prmean = prob_0_prmean

		# Define forgetting factors
		inv_lambda = 1/self.LAMBDA
		self.inv_lambda = inv_lambda

		# Define expert opinion (prior) model weight
		if self.expert_opinion == 1:
			expert_weight = 1/self.K
		elif self.expert_opinion == 2:
			expert_weight = 0
		if expert_weight == 0 and self.forgetting_method == 2:
			expert_weight = 1
		self.expert_weight = expert_weight

		return self

