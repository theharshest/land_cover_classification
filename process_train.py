import csv
import numpy as np
import pickle

arr = []

#File "labelled" is the file which has training data in the original format
with open('raw_train_data', 'rb') as f:
	reader = csv.reader(f)
	arr = [row for row in reader]

arr = np.array(arr)

#Getting rank, lat, lon info
info = arr[:,:2]
lat = arr[:,2]
lon = arr[:,3]

#Getting lables
labels = arr[:,4:18]

#All time series
val = arr[:,18:]

#Removing year 2000, i.e. one label value and 20 timestamp values
labels = labels[:,1:]
val = val[:,20:]

#To store processed train data
final = np.zeros((1,24))

k=0

for bigts in val:
	i = 0
	j = 23
	l = labels[k]
	t = 0
	while j<300:
		tmp = np.append(bigts[i:j], l[t])
		tmp.reshape(1,24)
		final = np.vstack((final, tmp))
		i = i+23
		j = j+23
		t = t+1
	k = k+1

f = open('train_data.pickle', 'wb')
pickle.dump(final, f)
f.close()
