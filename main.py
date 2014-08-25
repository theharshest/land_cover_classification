import pickle
import h5py
import numpy as np
import sys
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances
#import pylab as plt
import matplotlib.pyplot as plt
from operator import sub
from operator import mul
from mpl_toolkits.basemap import Basemap

class cluster:
	'''
	Cluster class holds a cluster:
	points - list of tuples, where each tuple represents a point,
		 each tuple is of the form (lat, lon, val) where val is a list of values
	stdev - standard deviation corresponding to that cluster
	'''
	def __init__(self, points, stdev):
		self.points = points
		self.stdev = stdev
	
	def update_points(self, points):
		self.points = points

	def update_stdev(self, stdev):
		self.stdev = stdev

def print_all(result):
	'''
	Prints the resultant set of clusters
	'''
	res = ""
	for i in result:
		res = res + "Cluster - {} points and {} standard deviation\n".format(len(i.points), i.stdev)

	return res

def plot_all(result):
	'''
	To plot all the clusters
	'''
        i = 1
        for c in result:
                fig = plt.figure(i)
                avg = []
		for p in c.points:
			avg.append(p[-1])
		avg = np.array(avg)
		std = np.std(avg, axis=0)
		avg = np.mean(avg, axis=0)
		avg1 = avg - 3*std
		avg2 = avg + 3*std
		plt.plot(avg1)
		plt.plot(avg)
		plt.plot(avg2)
                plt.savefig(str(i))
		plt.close()
                i = i+1

def apply_kmeans(result):
	'''
	Applies K-means to the cluster with maximum standard deviation with k=2
	Breaks that cluster and creates a new one with the remaining points
	'''
	#Index of cluster to break
	to_break = 0
	max_stdev = result[to_break].stdev

	#Finding out the cluster with maximum standard deviation
	for i in range(len(result)):
		if result[i].stdev > max_stdev:
			to_break = i
			max_stdev = result[i].stdev

	#Applying K Means to cluster with maximum standard deviation
	KM = MiniBatchKMeans(n_clusters=2, init='random', n_init=10)
	val = [i[2] for i in result[to_break].points]
	val = np.array(val)

	#Getting new cluster labels for the points
	res = KM.fit_predict(val)

	#Breaking the cluster
	tot_points = result[to_break].points

	points1 = [tot_points[i] for i in range(len(tot_points)) if res[i] == 0]
	points2 = [tot_points[i] for i in range(len(tot_points)) if res[i] == 1]

	stdev1 = get_stdev(points1)
	stdev2 = get_stdev(points2)

	#Updating the broken cluster
	result[to_break].update_points(points1)
	result[to_break].update_stdev(stdev1)

	#Creating a new cluster
	c1 = cluster(points2, stdev2)

	result.append(c1)

	return result

def get_stdev(points):
	'''
	To calculate standard deviation
	'''
	val = [i[2] for i in points]
	val = np.array(val)

	n = len(val)
	s = np.sum(val, axis=0)
	mu = s/n
	
	val_x = [map(sub, i, mu) for i in val]
	val_x = [map(mul, i, i) for i in val_x]
	val_x = np.sum(val_x, axis=1)
	val_x = np.sqrt(val_x)

	return sum(val_x)/np.sqrt(n)

def max_stdevs(result):
	'''
	To get maximum standard deviation
	'''
	res = result[0].stdev

	for i in result:
		res = max(res, i.stdev)

	return res

if __name__ == "__main__":
	#Getting input data in the form of .mat file 
	mat = h5py.File('data_unfiltered_h20v10_EVI_1000m_16day.mat')
	
	#Parsing data
	val = mat['data']
	dates = mat['dates']
	lat = mat['lat']
	lon = mat['lon']

	#Transposing to make vectors as columns
	val = np.array(val).transpose()
	lat = np.array(lat).transpose()
	lon = np.array(lon).transpose()

	#For fourier transformation
	#val = np.fft.fft(val, axis=1)
	#val = np.absolute(val)
	#val = (val - np.mean(val, axis=1)[:, np.newaxis]).clip(0)

	#Slicing timestamps from 16th to 269th
	val = val[:, 16:269]

	#Preprocessing the data (reshaping)
	ndays = 23
	take_columns = (val.shape[1]/ndays)*ndays
	latlon_extend = val.shape[1]/ndays
	val = val[:, :take_columns].T.reshape(-1, ndays, val.shape[0]).swapaxes(1, 2).reshape(-1, ndays)
	lat = np.tile(lat, (latlon_extend, 1))
	lon = np.tile(lon, (latlon_extend, 1))

	#Taking fourier transform
	val1 = np.fft.fft(val, axis=1)
	val1 = np.absolute(val1)

	#Classification of test set begins
	#KNN
	print "Running k-NN on test set.."
	val1 = val1[:,:3]
	f = open('train_knn.pickle')
	knn = pickle.load(f)
	f.close()

	result = knn.predict(val1)
	
	result = np.hstack((result.reshape(15840000,1), lat, lon))

	f = open('knn_results.pickle', 'wb')
	pickle.dump(result, f)
	f.close()

	raw_input("k-NN execution finished. Press enter to move to SVM.")

	#Linear SVM
	print "Running SVM on test set.."
	val1 = val1[:,:3]
	f = open('train_linearsvm.pickle')
	linearsvm = pickle.load(f)
	f.close()
	
	result = linearsvm.predict(val1)
	
	result = np.hstack((result.reshape(15840000,1), lat, lon))
	
	f = open('linearsvm_results.pickle', 'wb')
	pickle.dump(result, f)
	f.close()
	
	raw_input("SVM execution finished.")
  
	#Classification of test set ends

	val_freqs = val1
	#Keeping only first 5 coefficients and their corresponding conjugates and making every other coefficient 0
	val_freqs = [np.hstack((ts[:6],[0]*12,ts[-5:])) for ts in val_freqs]

	val1 = np.absolute(val1)

	n, m = val.shape

	val2 = np.float32(val1)
	val2[:,2] = val2[:,2]/val2[:,1]

	val = val[val2[:,2].argsort()]
	lat = lat[val2[:,2].argsort()]
	lon = lon[val2[:,2].argsort()]

	val = val[::-1]
	lat = lat[::-1]
	lon = lon[::-1]

	lat11 = lat[:n/2]
	lon11 = lon[:n/2]
	lat22 = lat[n/2:]
	lon22 = lon[n/2:]

	'''
	#Scatter plot of points
	fig = plt.figure("scatter")

	m = Basemap(llcrnrlon=20,llcrnrlat=-20,urcrnrlon=33,urcrnrlat=-10, resolution='c',projection='mill')

	x1, y1 = m(lon11, lat11)
	x2, y2 = m(lon22, lat22)

	x1 = x1[:10000]
	y1 = y1[:10000]
	x2 = x2[-10000:]
	y2 = y2[-10000:]

	m.plot(x1, y1, 'b.', alpha=0.5)
	m.plot(x2, y2, 'r.', alpha=0.5)

	plt.savefig("scatter")
	plt.close()

	fig = plt.figure("time_series")

	#Plots of individual time series
	for ts in val[:100]:
		plt.plot(ts, color='blue')
                plt.savefig("time_series")

	for ts in val[-100:]:
		plt.plot(ts, color='red')
                plt.savefig("time_series")

	plt.savefig("time_series")
	plt.close()
	'''

	i = 1
	j = 1

	#Plots of individual time series
	while j<len(val):
		plt.figure(i)
		plt.title(str(j))
		plt.plot(val[j], color='blue')
		j = j+10000
		plt.savefig(str(i))
		plt.close()
		i = i+1

	raw_input("hold")

	i = 5000
	for j in range(len(val1)-200):
		val2 = val1[j:j+1000]

		fig = plt.figure(i)
		plt.hist(val2, bins=20)
        	plt.savefig(str(i))
		plt.close()
		j = j+100
		i = i+1

	raw_input('--')

	#K-means fourier transformation
	print "Running bisecting k-means with fourier transformation..."

	#List to hold all the clusters
	result = []

	#Initial list of all points, a point is represented by a tuple
	tmp = zip(lat, lon, val)

	#Forming first cluster
	c1 = cluster(tmp, get_stdev(tmp))

	result.append(c1)

	f = open('output_w_fourier', 'w')

	while max_stdevs(result) > 800000:
		result = apply_kmeans(result)
		print print_all(result)
		print '--------------------'
		f.write(print_all(result))
		f.write('----------------------\n')

	plot_all(result)

	f.close()
