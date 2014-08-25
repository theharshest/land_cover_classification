from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import pickle
import numpy as np
import h5py

#Trained k-NN classifier
f = open('train_knn.pickle')
knn = pickle.load(f)
f.close()

f = open('fourier_dict.pickle')
fdict = pickle.load(f)
f.close()

d = []
d_fft = []

#Getting input data in the form of .mat file 
mat = h5py.File('data_unfiltered_h20v10_EVI_1000m_16day.mat')

#Parsing data
val = mat['data']
dates = mat['dates']
lat = mat['lat']
lon = mat['lon']

val = np.array(val).transpose()
lat = np.array(lat).transpose()
lon = np.array(lon).transpose()

val = val[:, 16:269]

#Preprocessing the data
ndays = 23
take_columns = (val.shape[1]/ndays)*ndays
latlon_extend = val.shape[1]/ndays
val = val[:, :take_columns].T.reshape(-1, ndays, val.shape[0]).swapaxes(1, 2).reshape(-1, ndays)
lat = np.tile(lat, (latlon_extend, 1)) 
lon = np.tile(lon, (latlon_extend, 1)) 

val_fft = np.fft.fft(val, axis=1)
val_fft = np.absolute(val_fft)

i=0
m,n = val.shape
while i<m:
	latlon = zip(lat[i:i+1440000,0], lon[i:i+1440000,0])
	tslatlon = zip(latlon, val[i:i+1440000,:])
	ts_fftlatlon = zip(latlon, val_fft[i:i+1440000,:])
	dt = dict(tslatlon)
	dt_fft = dict(ts_fftlatlon)
	d.append(dt)
	d_fft.append(dt_fft)
	i = i+1440000

yr = 0

z1 = []
z2 = []
o1 = []
o2 = []
t1 = []
t2 = []

def on_pick(event):
	global val
	global val_fft
	global knn
	global d
	global yr
	global z1, z2, o1, o2, t1, t2
	m = Basemap(projection='mill', llcrnrlat=-19.995856, urcrnrlat=-16, llcrnrlon=26, urcrnrlon=31.920095, resolution='l')
	point = event.artist
	ind = event.ind
	x, y = point.get_data()

	x, y = x[ind], y[ind]
	y, x =  m(x,y,inverse=True)
	s = "Time series for " + str(x) + "," + str(y)
	fig = plt.figure(s)
	plt.title(s)
	x = np.around(x, decimals=6)
	y = np.around(y, decimals=6)
	tpl = tuple((x, y))
	print "Point is ", x, y
	if tpl in z1:
		print "SVM class 0"
	elif tpl in o1:
		print "SVM class 1"
	elif tpl in t1:
		print "SVM class 2"
	else:
		print "none"
	
	if tpl in z2:
		print "k-NN class 0"
	elif tpl in o2:
		print "k-NN class 1"
	elif tpl in t2:
		print "k-NN class 2"
	else:
		print "none"
	
	plt.subplot2grid((2,2),(0, 0))
	plt.plot(d[yr-1][(x[0],y[0])])
	plt.xlabel("Time series for " + str(x) + str(y) + "(Point Clicked)")
	plt.subplot2grid((2,2),(0, 1))
	plt.plot(d_fft[yr-1][(x[0],y[0])])
	plt.xlabel("Fourier transform for " + str(x) + str(y) + "(Point Clicked)")
	resk = knn.kneighbors(d_fft[yr-1][(x[0],y[0])][:3], n_neighbors=1, return_distance=False)

	ind = resk[0][0]

	plt.subplot2grid((2,2),(1, 0))
	plt.plot(val[ind])
	plt.xlabel("Time series for 1-NN")

	plt.subplot2grid((2,2),(1, 1))
	plt.plot(val_fft[ind])
	plt.xlabel("Fourier transform for 1-NN")
	plt.show()

if __name__ == '__main__':
	#Getting classifiers results
	f1 = open('linearsvm_results.pickle')
	f2 = open('knn_results.pickle')
	predict1 = pickle.load(f1)
	predict2 = pickle.load(f2)
	f1.close()
	f2.close()

	m, n = predict1.shape

	points1 = []
	points2 = []

	i=0

	#Arranging data year-wise
	while i<m:
		points1.append(predict1[i:i+1440000,:])
		points2.append(predict2[i:i+1440000,:])
		i=i+1440000

	lbl = 1

	for p1, p2 in zip(points1,points2):
		yr = yr+1
		fig = plt.figure("Plot for year " + str(lbl))
		fig, axes = plt.subplots(nrows=1, ncols=2)

		#Initializing two basemaps for knn and svm
		m1 = Basemap(projection='mill', llcrnrlat=-19.995856, urcrnrlat=-16, llcrnrlon=26, urcrnrlon=31.920095, resolution='l', ax=axes.flat[0])
		m2 = Basemap(projection='mill', llcrnrlat=-19.995856, urcrnrlat=-16, llcrnrlon=26, urcrnrlon=31.920095, resolution='l', ax=axes.flat[1])

		#Drawing coastlines
		m1.drawcoastlines()
		m2.drawcoastlines()

		#Drawing latitudes from -20 to -10 with a gap of 1
		parallels = np.arange(-20.,-10.,1.)
		m1.drawparallels(parallels,labels=[False,True,True,False])
		m2.drawparallels(parallels,labels=[False,True,True,False])

		#Drawing longitudes from 20 to 32 with a gap of 1
		meridians = np.arange(20.,32.,1.)
		m1.drawmeridians(meridians,labels=[True,False,False,True])
		m2.drawmeridians(meridians,labels=[True,False,False,True])

		#Getting all points marked as 0 for ith year
		zeros1 = p1[np.where(p1[:,0] == 0)]
		zeros2 = p2[np.where(p2[:,0] == 0)]
		#Getting all points marked as 1 for ith year
		ones1 = p1[np.where(p1[:,0] == 1)]
		ones2 = p2[np.where(p2[:,0] == 1)]
		#Getting all points marked as 2 for ith year
		twos1 = p1[np.where(p1[:,0] == 2)]
		twos2 = p2[np.where(p2[:,0] == 2)]

		zx1, zy1 = m1(zeros1[:,2],zeros1[:,1])
		zx2, zy2 = m2(zeros2[:,2],zeros2[:,1])
		ox1, oy1 = m1(ones1[:,2],ones1[:,1])
		ox2, oy2 = m2(ones2[:,2],ones2[:,1])
		tx1, ty1 = m1(twos1[:,2],twos1[:,1])
		tx2, ty2 = m2(twos2[:,2],twos2[:,1])

		z1 = zip(zeros1[:,1],zeros1[:,2])
		o1 = zip(ones1[:,1],ones1[:,2])
		t1 = zip(twos1[:,1],twos1[:,2])
		z2 = zip(zeros2[:,1],zeros2[:,2])
		o2 = zip(ones2[:,1],ones2[:,2])
		t2 = zip(twos2[:,1],twos2[:,2])

		#Plotting unknown points with white (points with label as 0)
		m1.plot(zx1, zy1, 'w.', picker=2)
		#Plotting unknown points with blue (points with label as 1)
		m1.plot(ox1, oy1, 'b.', picker=2)
		#Plotting unknown points with yellow (points with label as 2)
		m1.plot(tx1, ty1, 'y.', picker=2)

		#Same plotting as above for k-NN
		m2.plot(zx2, zy2, 'w.', picker=2)
		m2.plot(ox2, oy2, 'b.', picker=2)
		m2.plot(tx2, ty2, 'y.', picker=2)

		cid = fig.canvas.mpl_connect('pick_event', on_pick)

		plt.show()

		plt.savefig("knn_" + str(lbl))
		plt.close()

		lbl = lbl+1
