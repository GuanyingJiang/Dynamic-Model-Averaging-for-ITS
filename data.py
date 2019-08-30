# coding=utf-8
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

from shapely.geometry import Point, Polygon
import geopandas as gpd
import folium

font = {'family':'normal','size':17}
matplotlib.rc('font',**font)
sns.set(font_scale=1.8)
sns.set_style('ticks')

SEED = 0
np.random.seed(SEED)
random.seed(SEED)

from branca.colormap import linear,ColorMap
import rpy2.robjects.lib.ggplot2 as ggplot2
import rpy2.robjects as r
from rpy2.robjects.packages import importr

geojsonio = importr('geojsonio')
leaflet = importr('leaflet')
mapview = importr('mapview')
openxlsx = importr('openxlsx')


# DATA PREPROCESSING
def preprocess_data(agg_file):
	t0 = time()
	cname = ['tripduration', 'starttime', 'stoptime', 'startStationId', 'startStationName',
				'startStationLatitude', 'startStationLongitude', 'endStationId', 'endStationName',
				'endStationLatitude', 'endStationLongitude', 'bikeid', 'usertype', 'birthyear', 'gender']

	months = ['201708', '201709', '201710', '201711', '201712', '201801', '201802', '201803',
				'201804', '201805', '201806', '201807', '201808', '201809', '201810']

	raw_data = pd.DataFrame()
	for m in months:
		data = pd.read_csv('citibike/' + m + '-citibike-tripdata.csv', skiprows=1, names=cname)
		## OD pairs

		data['starttime'] = [dtt.strptime(i[:19], '%Y-%m-%d %H:%M:%S') for i in data['starttime']]
		data['stoptime'] = [dtt.strptime(i[:19], '%Y-%m-%d %H:%M:%S') for i in data['stoptime']]

		data['startday'] = data['starttime'].dt.date
		data['starthour'] = data['starttime'].dt.hour
		data['startmonth'] = data['starttime'].dt.month

		raw_data = raw_data.append(data)
		print('total observations: %d.\ndataset: %s, observations: %d, seconds: %.4fs\n' 
				% (len(raw_data), m, len(data), time()-t0), data.head())

	raw_data.to_csv(agg_file)
	print('Finish preprocessing data. Used: %.4fs.\n' % (time()-t0))
	return raw_data

def data_subsets(agg_file):
	raw_data = pd.read_csv(agg_file)
	print('raw dataset: ', raw_data.shape)
	t0 = time()

	# all_trips2
	data1 = raw_data[raw_data.tripduration <= 45*3*60]

	birth_range1 = [1943,1963,1983,2003]
	birth_range2 = [1983,1988,1993,1998,2003]
	user_type = ['Subscriber','Customer']
	genders = [1,2]
	all_trips2 = pd.DataFrame()

	df_names = ['esub', 'ecus', 'emale', 'efemale',
				'msub', 'mcus', 'mmale', 'mfemale',
				'ysub', 'ycus', 'ymale', 'yfemale',
				'y4sub', 'y4cus', 'y4male', 'y4female',
				'y3sub', 'y3cus', 'y3male', 'y3female',
				'y2sub', 'y2cus', 'y2male', 'y2female',
				'y1sub', 'y1cus', 'y1male', 'y1female',
				'gd1sub', 'gd2sub', 'gd1cus', 'gd2cus']
	i = 0
	for br in [birth_range1, birth_range2]:
		for lbr in range(len(br)-1):
			for d in user_type:
				temp = data1[data1.birthyear.isin(range(br[lbr], br[lbr+1])) & (data1.usertype == d)]
				dates = pd.crosstab(index=temp.startday, columns='count')  # nan->0
				all_trips2[df_names[i]] = dates['count']
				i += 1
			for d in genders:
				temp = data1[data1.birthyear.isin(range(br[lbr], br[lbr+1])) & (data1.gender == d)]
				dates = pd.crosstab(index=temp.startday, columns='count')  # nan->0
				all_trips2[df_names[i]] = dates['count']
				i += 1
		print('cost %.4fs ' %(time()-t0), all_trips2.shape)

	for ut in user_type:
		for gd in genders:
			temp = data1[data1.birthyear.isin(range(1983,2003)) & (data1.usertype == ut) & (data1.gender == gd)]
			dates = pd.crosstab(index=temp.startday, columns='count')  # nan->0
			all_trips2[df_names[i]] = dates['count']
			i += 1
	all_trips2 = all_trips2[153:]
	print('cost %.4fs ' %(time()-t0), all_trips2.shape)

	# all_trips1
	data1 = raw_data[raw_data.tripduration <= 45*3*60]
	df_names = ['elderly','middle','young','young4','young3','young2','young1']
	i = 0
	all_trips1 = pd.DataFrame()

	for br in [birth_range1, birth_range2]:
		for lbr in range(len(br)-1):
			temp = data1[data1.birthyear.isin(range(br[lbr], br[lbr+1])) & (data1.usertype == 'Subscriber')]
			dates = pd.crosstab(index=temp.startday, columns='count')  # nan->0
			all_trips1[df_names[i]] = dates['count']
			i += 1
	all_trips1 = all_trips1[:-61]
	print('cost %.4fs ' %(time()-t0), all_trips1.shape)

	for i,j in enumerate([all_trips1, all_trips2]):
		sleep(1)
		writer = pd.ExcelWriter('data/trips' + str(i+1) + '.xlsx')
		for idx in j.columns:
			print(idx)
			j[idx].to_excel(writer, sheet_name=idx, index=False)
		writer.save()

	print('Finish splitting data subsets.')
	return all_trips1, all_trips2

def trips_stats():
	allstats = pd.DataFrame()
	for ii in (1,2):
		file = pd.ExcelFile('data/trips' + str(ii) + '.xlsx')
		for sheet in file.sheet_names:
			data = file.parse(sheet)[sheet][31:]
			stats = np.matrix([data.sum(), data.mean(), data.std(), data.max(), data.min()])
			stats_df = pd.DataFrame(stats, 
				columns=['count', 'mean', 'std', 'max', 'min'], index=[sheet])
			allstats = allstats.append(stats_df)
	allstats.to_excel('data/all_stats.xlsx')
	return allstats

def weather_corr():
	writer = pd.ExcelWriter('data/corr.xlsx')

	for i,sheet in enumerate(('stat2', 'stat')):
		weather = pd.read_excel('data/weather.xlsx', sheet_name=sheet)
		weather.AvgTemp, weather.AvgDew, weather.AvgPress = weather.AvgTemp.diff(), weather.AvgDew.diff(), weather.AvgPress.diff()
		corr = weather.corr()
		idx = weather.columns.tolist()
		corr.to_excel(writer, sheet_name=sheet, index=idx, columns=idx)

		seasons = [0,91,181,273,365]
		snames = ['fall', 'winter', 'spring', 'summer']
		for ii in range(4):
			temp = weather[seasons[ii]:seasons[ii+1]]
			temp_stat = pd.DataFrame({'mean': temp.mean(), 'std': temp.std()})
			temp_stat.to_excel(writer, sheet_name = snames[ii] + str(i+1))
	writer.save()

	weather = np.nan_to_num(np.array(weather))
	months = [0,30,61,91,122,153,181,212,242,273,303,334,365]
	corr_mat = np.zeros((12,15))
	k = -1
	for i in range(5):
		for j in range(i+1,6):
			k += 1
			for m in range(12):
				corr_mat[m,k] = np.corrcoef(weather[months[m]:months[m+1],i], weather[months[m]:months[m+1],j])[0,1]

	corr_mat = pd.DataFrame(corr_mat, columns=['Precip_Temp', 'Precip_Dew', 'Precip_Humid', 'Precip_Wind',
					'Precip_Press','Temp_Dew', 'Temp_Humid', 'Temp_Wind', 'Temp_Press', 'Dew_Humid', 'Dew_Wind',
					'Dew_Press', 'Humid_Wind', 'Humid_Press', 'Wind_Press'], index=['Sep.', 'Oct.', 'Nov.',
					'Dec.', 'Jan.', 'Feb.', 'Mar.', 'Apr.', 'May', 'Jun.', 'Jul.', 'Aug.'])

	fig = plt.figure(figsize=(12,14))
	sns.heatmap(corr_mat.T, annot=False, square=True, cmap='coolwarm', cbar=True, linewidths=0.1, linecolor='white', 
						vmin=-1, vmax=1, center=0, cbar_kws={'shrink':0.6})
	plt.tight_layout()
	plt.savefig('fig/corr_4s.pdf')
	return

def seasonal_dynamics():
	census = gpd.read_file('data/2010CensusTracts.geojson')
	print(census.columns.values, census.shape, '\n\n')

	raw_data = pd.read_csv('data/all_tripdata.csv')
	data = raw_data[(raw_data.tripduration <= 45*3*60) & (raw_data.usertype == 'Subscriber')
					& raw_data.birthyear.isin(range(1983,2003))]

	keys = ['startStationLatitude', 'startStationLongitude']
	unique_stn = data.drop_duplicates(subset=keys, keep='first')
	unique_stn['geo'] = np.zeros(unique_stn.shape[0])
	unique_stn.dropna(subset=keys, inplace=True)
	unique_stn.reset_index(drop=True, inplace=True)

	for u in range(len(unique_stn)):
		point = Point(unique_stn['startStationLongitude'][u], unique_stn['startStationLatitude'][u])
		for c in range(len(census)):
			if census['geometry'][c].contains(point):
				unique_stn['geo'][u] = int(c)  # +1?
				break

	unique_stn = unique_stn[['geo','startStationId','startStationName'] + keys]
	writer = pd.ExcelWriter('data/unique_stn.xlsx')
	unique_stn.to_excel(writer, sheet_name='unique_station')

	mts = [[9,10,11], [12,1,2], [3,4,5], [6,7,8]]
	snames = ['Fall', 'Winter', 'Spring', 'Summer']

	for ii,kk in enumerate(mts):
		sub_data = data[data.startmonth.isin(kk)]
		sub_data['year'] = [dtt.strptime(ii[:4],'%Y') for ii in sub_data['startday']]

		if ii in (0,3):
			sub_data = sub_data[sub_data.year == ('2017' if ii == 0 else '2018')]
		print(sub_data.startday.unique())

		cnt_stn = pd.crosstab(index=sub_data.startStationId, columns='count')
		station = pd.merge(unique_stn, cnt_stn, how='left', on='startStationId')
		ustations = pd.pivot_table(station, values='count', index='geo', aggfunc=np.sum, fill_value=0)

		ustations.reset_index(level=0, inplace=True)
		ustations.to_excel(writer, sheet_name=snames[ii])

	writer.save()
	return

def map_seasonal_dynamics():
	r.r('''
		state = geojsonio::geojson_read('data/2010CensusTracts.geojson', what='sp', parse=True)
		ustation = read.xlsx('data/unique_stn.xlsx', sheet='unique_station', colNames=TRUE)
		bins = c(0, 1000, 2000, 5000, 10000, 20000, 50000, Inf)

		for (ii in c('Fall', 'Winter', 'Spring', 'Summer')){
			subdata = read.xlsx('data/unique_stn.xlsx', sheet=ii, colNames=TRUE)
			subdata$geo = subdata$geo + 1
			states = state[as.vector(subdata$geo),]

			pal = colorBin('RdPu', domain=subdata, bins=bins)  #YlOrRd
			m = leaflet(states) %>%
				addProviderTiles(providers$CartoDB.Positron) %>%  #ALTER!!!
				setView(-73.95, 40.73, 12)

			map = m %>%
					addPolygons(fillColor = ~pal(as.matrix(subdata)), 
												weight = 2, 
												opacity = 1, 
												color = 'white', 
												dashArray = '1', 
												fillOpacity = 0.7) %>%
					addCircleMarkers(~as.numeric(ustation$startStationLongitude),
									~as.numeric(ustation$startStationLatitude),	
									radius=1,
									color = 'brown',  ##~pal(xx)
									stroke = F, 
									fillOpacity = 1) %>%
					addLegend(pal = pal, 
								values = ~subdata, 
								opacity = 0.8,
								title = ii,
								position = 'topleft')

			mapshot(map, file=paste('map_', ii, '.pdf', sep=''), url=paste('map_', ii, '.html', sep=''))
			print(ii)}
			''')
	return


# if __name__ == '__main__':
# 	filename = 'data/all_tripdata.csv'
# 	raw_data = preprocess_data(filename)
# 	all_trips1, all_trips2 = data_subsets(filename)
# 	trips_stats()
# 	weather_corr()
# 	seasonal_dynamics()
# 	map_seasonal_dynamics()
