import h5py
import numpy as np
import pickle

#Input
mat = h5py.File('data_unfiltered_h20v10_EVI_1000m_16day.mat')
val = mat['data']
dates = mat['dates']
lat = mat['lat']
lon = mat['lon']

val = val[:, 16:269]

#Preprocessing the data (reshaping)
ndays = 23
take_columns = (val.shape[1]/ndays)*ndays
latlon_extend = val.shape[1]/ndays
val = val[:, :take_columns].T.reshape(-1, ndays, val.shape[0]).swapaxes(1, 2).reshape(-1, ndays)
lat = np.tile(lat, (latlon_extend, 1)) 
lon = np.tile(lon, (latlon_extend, 1)) 

val = np.fft.fft(val, axis=1)
val = np.absolute(val)

#Taking fourier transform
val1 = np.fft.fft(val, axis=1)
val1 = np.absolute(val1)

#Creating dict
val2 = val1[:,:3]

res = {}

ts_fourier = zip(val, val1)

res = {tuple(k): v for (k, v) in zip(val2, ts_fourier)}

f = open('fourier_dict.pickle', 'wb')
pickle.dump(res, f)
f.close()

